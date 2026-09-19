"""Tests for translating a natural-language question into a PubMed term.

This exists because of a failure found against the live API, not in a mock. Sent
verbatim, "What is the role of the gut microbiome in colorectal cancer?" returns
zero papers, because NCBI's automatic term mapping reads "is" as an author name
and ANDs ``is[Author]`` with everything else. The same question with question
words removed returns thousands.

So the property under test is not tidiness. It is that words which silently
corrupt a query are removed, while the words that carry meaning, including
hyphenated identifiers, survive exactly as written.
"""

import pytest

from bio_research_copilot.adapters.pubmed import (
    QUESTION_WORDS,
    narrow_search_term,
    to_search_term,
)


class TestRemovesHazardousWords:
    def test_drops_the_word_that_maps_to_an_author_field(self) -> None:
        """The specific cause of the live failure."""
        assert "is" not in to_search_term("What is the cause of X?").lower().split()

    def test_drops_question_words_and_keeps_content(self) -> None:
        assert (
            to_search_term("What is the role of the gut microbiome in colorectal cancer?")
            == "role gut microbiome colorectal cancer"
        )

    @pytest.mark.parametrize(
        "question",
        [
            "How do statins work?",
            "Why does aspirin reduce inflammation?",
            "Which kinases phosphorylate tau?",
        ],
    )
    def test_no_question_word_survives(self, question: str) -> None:
        terms = {word.lower() for word in to_search_term(question).split()}
        assert not (terms & QUESTION_WORDS)

    def test_the_word_list_is_ascii(self) -> None:
        """Guards against a stray non-English entry silently never matching."""
        assert all(word.isascii() and word.islower() for word in QUESTION_WORDS)


class TestPreservesMeaning:
    @pytest.mark.parametrize(
        ("question", "kept"),
        [
            ("How do PD-1 inhibitors work?", "PD-1"),
            ("What causes long COVID-19 symptoms?", "COVID-19"),
            ("Is CRISPR/Cas9 accurate?", "CRISPR/Cas9"),
            ("What is the role of TP53 in cancer?", "TP53"),
        ],
    )
    def test_identifiers_survive_intact(self, question: str, kept: str) -> None:
        """A tokeniser splitting on punctuation would break these into parts."""
        assert kept in to_search_term(question).split()

    def test_case_is_preserved(self) -> None:
        assert "GLP-1" in to_search_term("How do GLP-1 agonists cause weight loss?")

    def test_strips_only_edge_punctuation(self) -> None:
        assert to_search_term("Does metformin, or insulin, work better?") == (
            "metformin insulin work better"
        )

    def test_order_is_preserved(self) -> None:
        assert to_search_term("What causes tau phosphorylation?") == "causes tau phosphorylation"


class TestFallback:
    def test_a_question_of_only_stopwords_is_sent_as_written(self) -> None:
        """An empty term is worse than an imperfect one."""
        assert to_search_term("What is it?") == "What is it?"

    def test_blank_input_stays_blank(self) -> None:
        assert to_search_term("   ") == ""

    def test_a_plain_keyword_query_is_unchanged(self) -> None:
        assert to_search_term("base editing off-target") == "base editing off-target"


class TestConversationalFraming:
    """Regression: a question phrased as a request returned nothing at all.

    NCBI ANDs terms it does not recognise as fields, so one conversational word
    is enough to reduce a query from hundreds of papers to zero. "Tell me about
    MYBPC3" matched nothing while "MYBPC3" matched several hundred.
    """

    @pytest.mark.parametrize(
        "question",
        [
            "Tell me about MYBPC3",
            "please describe MYBPC3",
            "give me an overview of MYBPC3",
            "Can you explain MYBPC3 to me?",
            "I want to know about MYBPC3",
            "summarise MYBPC3 for me",
            "provide details regarding MYBPC3",
        ],
    )
    def test_framing_is_stripped_down_to_the_subject(self, question: str) -> None:
        assert to_search_term(question) == "MYBPC3"

    def test_meaningful_words_are_not_treated_as_framing(self) -> None:
        """Dropping these would change what was asked."""
        term = to_search_term("What are the effects and causes of heart failure?")
        for kept in ("effects", "causes", "heart", "failure"):
            assert kept in term.split()


class TestNarrowing:
    """The fallback for framing words the list does not know about."""

    @pytest.mark.parametrize(
        ("term", "expected"),
        [
            ("elucidate MYBPC3 pathogenicity", "MYBPC3"),
            ("kindly ascertain PD-1 blockade outcomes", "PD-1"),
            ("ramble concerning COVID-19 transmission", "COVID-19"),
        ],
    )
    def test_keeps_the_distinctive_identifier(self, term: str, expected: str) -> None:
        """A gene, protein or acronym carries a digit or an interior capital."""
        assert narrow_search_term(term) == expected

    def test_keeps_several_identifiers_when_present(self) -> None:
        assert narrow_search_term("compare TP53 and BRCA1 variants") == "TP53 BRCA1"

    def test_falls_back_to_the_longest_words(self) -> None:
        narrowed = narrow_search_term("ramble regarding inflammation of pancreatic tissue")
        assert "inflammation" in narrowed
        assert "of" not in narrowed.split()

    def test_preserves_the_original_word_order(self) -> None:
        assert narrow_search_term("compare TP53 and BRCA1 variants") == "TP53 BRCA1"

    def test_a_single_token_cannot_be_narrowed(self) -> None:
        assert narrow_search_term("MYBPC3") == ""

    def test_nothing_to_drop_returns_empty(self) -> None:
        """Signals to the caller that retrying would repeat the same query."""
        assert narrow_search_term("causes heart failure") == ""

    def test_empty_input(self) -> None:
        assert narrow_search_term("") == ""
