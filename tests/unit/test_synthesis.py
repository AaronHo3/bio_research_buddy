"""Tests for extractive synthesis and evidence budgeting.

The central property is that a claim is always text that already existed in a
cited passage. That is what makes this baseline faithful by construction, so it
is asserted directly rather than inferred.
"""

import hashlib

import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, Source
from bio_research_copilot.synthesis import (
    ExtractiveSynthesizer,
    select_evidence,
)

QUESTION = "What causes off-target editing by base editors?"


def evidence(chunk: str, text: str, rank: int) -> RetrievedEvidence:
    pmid, position = chunk.split(":")
    return RetrievedEvidence(
        passage=Passage(
            source=Source.PUBMED,
            source_id=pmid,
            chunk_id=f"pubmed:{chunk}",
            position=int(position),
            section="abstract",
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            strategy="test",
        ),
        rank=rank,
        score=1.0 / rank,
        method="test",
    )


class TestExtractiveSynthesizer:
    def test_every_claim_is_copied_verbatim_from_its_cited_passage(self) -> None:
        items = [
            evidence("1:0", "Deaminase activity causes off-target editing. Cells grew well.", 1),
            evidence("2:0", "Guide RNA design affects editing specificity in base editors.", 2),
        ]
        citations = ExtractiveSynthesizer().synthesize(QUESTION, items)

        assert citations
        by_chunk = {e.passage.chunk_id: e.passage.text for e in items}
        for citation in citations:
            source_text = by_chunk[citation.evidence_ids[0]]
            assert citation.claim in source_text

    def test_picks_the_sentence_that_shares_most_query_terms(self) -> None:
        text = "Cells were cultured overnight. Off-target editing arises from deaminase activity."
        citations = ExtractiveSynthesizer().synthesize(QUESTION, [evidence("1:0", text, 1)])
        assert citations[0].claim == "Off-target editing arises from deaminase activity."

    def test_a_passage_with_no_overlap_yields_no_claim(self) -> None:
        items = [evidence("1:0", "Zebrafish were housed at twenty-eight degrees.", 1)]
        assert ExtractiveSynthesizer().synthesize(QUESTION, items) == []

    def test_claims_follow_evidence_rank(self) -> None:
        items = [
            evidence("2:0", "Base editors show off-target effects in vivo.", 2),
            evidence("1:0", "Off-target editing depends on deaminase choice.", 1),
        ]
        citations = ExtractiveSynthesizer().synthesize(QUESTION, items)
        assert [c.evidence_ids[0] for c in citations] == ["pubmed:1:0", "pubmed:2:0"]

    def test_each_claim_cites_exactly_the_passage_it_came_from(self) -> None:
        items = [
            evidence("1:0", "Off-target editing depends on deaminase choice.", 1),
            evidence("2:0", "Base editors show off-target effects in vivo.", 2),
        ]
        citations = ExtractiveSynthesizer().synthesize(QUESTION, items)
        assert all(len(c.evidence_ids) == 1 for c in citations)
        assert [c.evidence_ids[0] for c in citations] == ["pubmed:1:0", "pubmed:2:0"]

    def test_respects_the_claim_limit(self) -> None:
        items = [evidence(f"{i}:0", f"Off-target editing note {i}.", i) for i in range(1, 9)]
        citations = ExtractiveSynthesizer(max_claims=3).synthesize(QUESTION, items)
        assert len(citations) == 3

    def test_identical_sentences_are_not_repeated(self) -> None:
        text = "Off-target editing depends on deaminase choice."
        items = [evidence("1:0", text, 1), evidence("2:0", text, 2)]
        assert len(ExtractiveSynthesizer().synthesize(QUESTION, items)) == 1

    def test_no_evidence_yields_no_claims(self) -> None:
        assert ExtractiveSynthesizer().synthesize(QUESTION, []) == []

    def test_a_question_of_only_stopwords_yields_no_claims(self) -> None:
        items = [evidence("1:0", "Off-target editing depends on deaminase choice.", 1)]
        assert ExtractiveSynthesizer().synthesize("what is the", items) == []

    def test_a_stricter_overlap_requirement_rejects_weak_matches(self) -> None:
        items = [evidence("1:0", "Editing was performed.", 1)]
        assert ExtractiveSynthesizer(min_overlap=1).synthesize(QUESTION, items)
        assert ExtractiveSynthesizer(min_overlap=3).synthesize(QUESTION, items) == []

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_claims": 0}, "max_claims must be at least 1"),
            ({"min_overlap": 0}, "min_overlap must be at least 1"),
        ],
    )
    def test_invalid_configuration(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ExtractiveSynthesizer(**kwargs)


class TestSelectEvidence:
    def test_keeps_the_highest_ranked_within_budget(self) -> None:
        items = [evidence(f"{i}:0", "x" * 100, i) for i in range(1, 6)]
        selected = select_evidence(items, char_budget=250)
        assert [e.rank for e in selected] == [1, 2]

    def test_returns_everything_when_the_budget_allows(self) -> None:
        items = [evidence(f"{i}:0", "x" * 10, i) for i in range(1, 4)]
        assert len(select_evidence(items, char_budget=1000)) == 3

    def test_always_returns_at_least_one_item(self) -> None:
        """A single oversized passage must not become a silent abstention."""
        items = [evidence("1:0", "x" * 5000, 1)]
        assert len(select_evidence(items, char_budget=10)) == 1

    def test_orders_by_rank_regardless_of_input_order(self) -> None:
        items = [evidence("3:0", "x", 3), evidence("1:0", "x", 1), evidence("2:0", "x", 2)]
        assert [e.rank for e in select_evidence(items)] == [1, 2, 3]

    def test_empty_input(self) -> None:
        assert select_evidence([]) == []

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="char_budget must be at least 1"):
            select_evidence([], char_budget=0)
