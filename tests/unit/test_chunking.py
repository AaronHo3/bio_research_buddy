"""Tests for the section chunker: deterministic ids, section detection, no invented text."""

import hashlib
from pathlib import Path

import pytest

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.domain import Paper, SectionChunker, Source
from bio_research_copilot.domain.chunking import FullTextChunker, split_into_windows

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))


def paper(abstract: str | None, pmid: str = "1") -> Paper:
    return Paper(
        source=Source.PUBMED,
        source_id=pmid,
        title="A title.",
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def test_title_is_always_the_first_passage() -> None:
    passages = SectionChunker().split(paper(abstract=None))
    assert len(passages) == 1
    assert passages[0].section == "title"
    assert passages[0].text == "A title."
    assert passages[0].position == 0


def test_plain_abstract_is_one_passage() -> None:
    passages = SectionChunker().split(paper("First sentence. Second sentence."))
    assert [(p.section, p.text) for p in passages] == [
        ("title", "A title."),
        ("abstract", "First sentence. Second sentence."),
    ]


def test_structured_abstract_splits_at_labels() -> None:
    text = "BACKGROUND: Context here.\n\nMETHODS AND RESULTS: We measured.\n\nCONCLUSIONS: Done."
    passages = SectionChunker().split(paper(text))
    assert [(p.section, p.text) for p in passages[1:]] == [
        ("BACKGROUND", "Context here."),
        ("METHODS AND RESULTS", "We measured."),
        ("CONCLUSIONS", "Done."),
    ]


def test_sentence_case_prefix_is_not_mistaken_for_a_label() -> None:
    passages = SectionChunker().split(paper("Note: this is a plain abstract."))
    assert passages[1].section == "abstract"
    assert passages[1].text == "Note: this is a plain abstract."


def test_blank_parts_are_dropped_and_text_is_stripped() -> None:
    passages = SectionChunker().split(paper("  Body.  \n\n   \n\nMore."))
    assert [p.text for p in passages[1:]] == ["Body.", "More."]


def test_ids_are_deterministic_and_positional() -> None:
    first = SectionChunker().split(paper("X.", pmid="99"))
    second = SectionChunker().split(paper("X.", pmid="99"))
    assert first == second
    assert [p.chunk_id for p in first] == ["pubmed:99:0", "pubmed:99:1"]


def test_content_hash_is_sha256_of_text() -> None:
    passage = SectionChunker().split(paper("Exact text."))[1]
    assert passage.content_hash == hashlib.sha256(b"Exact text.").hexdigest()


def test_changed_text_keeps_id_but_changes_hash() -> None:
    before = SectionChunker().split(paper("Version one."))[1]
    after = SectionChunker().split(paper("Version two."))[1]
    assert before.chunk_id == after.chunk_id
    assert before.content_hash != after.content_hash


def test_strategy_name_is_recorded() -> None:
    assert all(p.strategy == "section-v1" for p in SectionChunker().split(RECORDED[0]))


def test_recorded_fixture_yields_title_plus_abstract_each() -> None:
    for original in RECORDED:
        passages = SectionChunker().split(original)
        assert len(passages) == 2
        assert passages[0].text == original.title
        assert passages[1].text == original.abstract
        assert all(p.source_id == original.source_id for p in passages)


class TestSplitIntoWindows:
    """Windowing full text.

    A body is far too long to be one citable passage, so it is cut into
    overlapping windows. The properties that matter are that nothing is lost at
    a boundary and that a cut lands at a sentence end where one is available.
    """

    def test_short_text_is_one_window(self) -> None:
        assert split_into_windows("Short enough.", size=100, overlap=10) == ["Short enough."]

    def test_long_text_is_cut_into_several(self) -> None:
        text = " ".join(f"Sentence number {n} of the body." for n in range(80))
        windows = split_into_windows(text, size=300, overlap=40)
        assert len(windows) > 3
        assert all(len(w) <= 300 for w in windows)

    def test_windows_overlap_so_a_boundary_sentence_survives_whole(self) -> None:
        text = " ".join(f"Item {n} appears here." for n in range(60))
        windows = split_into_windows(text, size=200, overlap=60)
        joined = " ".join(windows)
        for n in range(60):
            assert f"Item {n} appears here." in joined

    def test_prefers_to_cut_at_a_sentence_end(self) -> None:
        text = "First sentence is reasonably long here. " * 12
        windows = split_into_windows(text, size=220, overlap=30)
        assert windows[0].endswith(".")

    def test_whitespace_is_normalised(self) -> None:
        windows = split_into_windows("  lots   of\n\n space  ", size=100, overlap=0)
        assert windows == ["lots of space"]

    def test_empty_text_yields_nothing(self) -> None:
        assert split_into_windows("   ", size=100, overlap=10) == []

    def test_terminates_on_pathological_overlap(self) -> None:
        """A window that never advances would hang the request."""
        text = "x" * 5000
        windows = split_into_windows(text, size=100, overlap=99)
        assert len(windows) < 5000

    @pytest.mark.parametrize(
        ("size", "overlap", "message"),
        [
            (0, 0, "size must be at least 1"),
            (10, 10, "overlap must be at least 0 and smaller than size"),
            (10, 20, "overlap must be at least 0 and smaller than size"),
        ],
    )
    def test_invalid_configuration(self, size: int, overlap: int, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            split_into_windows("text", size=size, overlap=overlap)


class TestFullTextChunker:
    def test_without_full_text_it_matches_the_abstract_chunker(self) -> None:
        article = paper(abstract="A plain abstract about base editing accuracy.")
        assert [p.text for p in FullTextChunker().split(article)] == [
            p.text for p in SectionChunker().split(article)
        ]

    def test_full_text_passages_follow_the_abstract_ones(self) -> None:
        article = paper(abstract="An abstract.")
        chunker = FullTextChunker()
        base = len(chunker.split(article))
        passages = chunker.split_full_text(article, [("Results", "A finding. " * 40)])
        assert len(passages) > base
        assert [p.position for p in passages] == list(range(len(passages)))

    def test_chunk_ids_are_unique_and_deterministic(self) -> None:
        """A repeated id would silently overwrite evidence in the citation map."""
        article = paper(abstract="An abstract.")
        sections = [("Introduction", "Intro text. " * 40), ("Results", "Result text. " * 40)]
        first = FullTextChunker().split_full_text(article, sections)
        second = FullTextChunker().split_full_text(article, sections)
        ids = [p.chunk_id for p in first]
        assert len(ids) == len(set(ids))
        assert ids == [p.chunk_id for p in second]

    def test_the_section_label_is_carried_onto_every_piece(self) -> None:
        """A claim should be able to say where in the paper it came from."""
        article = paper(abstract="An abstract.")
        long_section = "Method detail number one. " * 150  # comfortably over one window
        passages = FullTextChunker().split_full_text(article, [("Methods", long_section)])
        method_passages = [p for p in passages if p.section == "Methods"]
        assert len(method_passages) > 1

    def test_records_its_own_strategy_name_on_full_text(self) -> None:
        article = paper(abstract="An abstract.")
        passages = FullTextChunker().split_full_text(article, [("Results", "Text. " * 60)])
        assert {p.strategy for p in passages if p.section == "Results"} == {"fulltext-v1"}

    def test_content_hash_tracks_the_text(self) -> None:
        article = paper(abstract="An abstract.")
        passages = FullTextChunker().split_full_text(article, [("Results", "Text. " * 60)])
        for passage in passages:
            assert passage.content_hash == hashlib.sha256(passage.text.encode()).hexdigest()

    def test_an_empty_section_list_leaves_only_the_abstract(self) -> None:
        article = paper(abstract="An abstract.")
        chunker = FullTextChunker()
        assert len(chunker.split_full_text(article, [])) == len(chunker.split(article))
