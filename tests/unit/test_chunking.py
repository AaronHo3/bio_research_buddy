"""Tests for the section chunker: deterministic ids, section detection, no invented text."""

import hashlib
from pathlib import Path

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.domain import Paper, SectionChunker, Source

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
