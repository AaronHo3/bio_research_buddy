"""Tests for the PubMed Central adapter.

Absence is the normal case here, so most of these assert what happens when full
text is *not* available: an article with no body, a link type that means the
opposite of what it looks like, a batched response that cannot be attributed.
Getting any of those wrong attaches text to the wrong paper, which is worse than
returning nothing.
"""

from pathlib import Path

import pytest

from bio_research_copilot.adapters.literature import LiteratureSourceParseError
from bio_research_copilot.adapters.pmc import (
    MIN_SECTION_CHARS,
    PMC_LINK_NAME,
    FullText,
    parse_pmc_links,
    parse_pmc_xml,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "pmc" / "efetch_pmc_sample.xml"
SAMPLE = FIXTURE.read_text(encoding="utf-8")


def linkset(pmid: str, links: list[str], name: str = PMC_LINK_NAME) -> dict[str, object]:
    return {"ids": [pmid], "linksetdbs": [{"linkname": name, "links": links}]}


class TestParseLinks:
    def test_maps_each_pmid_to_its_pmcid(self) -> None:
        payload = {"linksets": [linkset("1", ["100"]), linkset("2", ["200"])]}
        assert parse_pmc_links(payload) == {"1": "100", "2": "200"}

    def test_ignores_the_reverse_citation_link_type(self) -> None:
        """``pubmed_pmc_refs`` is papers citing this one, not this one's text."""
        payload = {"linksets": [linkset("1", ["999"], name="pubmed_pmc_refs")]}
        assert parse_pmc_links(payload) == {}

    def test_takes_only_the_pmc_link_when_both_are_present(self) -> None:
        payload = {
            "linksets": [
                {
                    "ids": ["1"],
                    "linksetdbs": [
                        {"linkname": "pubmed_pmc", "links": ["100"]},
                        {"linkname": "pubmed_pmc_refs", "links": ["900", "901"]},
                    ],
                }
            ]
        }
        assert parse_pmc_links(payload) == {"1": "100"}

    def test_a_batched_linkset_is_skipped_rather_than_guessed(self) -> None:
        """Comma-joined ids come back undifferentiated; attributing them invents data."""
        payload = {
            "linksets": [
                {
                    "ids": ["1", "2", "3"],
                    "linksetdbs": [{"linkname": PMC_LINK_NAME, "links": ["100", "200"]}],
                }
            ]
        }
        assert parse_pmc_links(payload) == {}

    def test_a_paper_with_no_pmc_record_is_absent(self) -> None:
        payload = {"linksets": [linkset("1", ["100"]), {"ids": ["2"], "linksetdbs": []}]}
        assert parse_pmc_links(payload) == {"1": "100"}

    def test_no_linksets_is_an_empty_mapping(self) -> None:
        assert parse_pmc_links({"linksets": []}) == {}

    @pytest.mark.parametrize("payload", ["not a dict", 42, {"nope": 1}])
    def test_an_unexpected_shape_is_an_error(self, payload: object) -> None:
        with pytest.raises(LiteratureSourceParseError):
            parse_pmc_links(payload)


class TestParseFullText:
    def test_returns_only_articles_that_have_a_body(self) -> None:
        """A publisher that blocks download sends metadata and no body."""
        assert [t.pmid for t in parse_pmc_xml(SAMPLE)] == ["30000001", "30000003"]

    def test_strips_the_pmc_prefix_from_the_identifier(self) -> None:
        """elink and efetch both take bare digits."""
        assert parse_pmc_xml(SAMPLE)[0].pmcid == "7000001"

    def test_keeps_section_headings_as_labels(self) -> None:
        labels = [s.label for s in parse_pmc_xml(SAMPLE)[0].sections]
        assert labels == ["Introduction", "Results", "Editing window"]

    def test_captures_nested_subsections(self) -> None:
        text = _section(parse_pmc_xml(SAMPLE)[0], "Editing window")
        assert "bystander edits" in text

    def test_drops_reference_lists_and_acknowledgements(self) -> None:
        """Citation strings would dilute a term-based ranking without ever being quotable."""
        labels = {s.label.lower() for s in parse_pmc_xml(SAMPLE)[0].sections}
        assert "references" not in labels
        assert "acknowledgements" not in labels

    def test_drops_tables_and_figure_captions(self) -> None:
        results = _section(parse_pmc_xml(SAMPLE)[0], "Results")
        assert "Figure caption text" not in results
        assert "99" not in results

    def test_drops_sections_too_short_to_be_evidence(self) -> None:
        labels = {s.label for s in parse_pmc_xml(SAMPLE)[0].sections}
        assert "Note" not in labels

    def test_a_section_title_is_not_repeated_in_its_text(self) -> None:
        assert not _section(parse_pmc_xml(SAMPLE)[0], "Introduction").startswith("Introduction")

    def test_an_article_without_sections_becomes_one_body_passage(self) -> None:
        article = parse_pmc_xml(SAMPLE)[1]
        assert [s.label for s in article.sections] == ["body"]
        assert "loose paragraphs" in article.sections[0].text

    def test_reports_total_characters(self) -> None:
        article = parse_pmc_xml(SAMPLE)[0]
        assert article.characters == sum(len(s.text) for s in article.sections)
        assert article.characters > MIN_SECTION_CHARS

    def test_whitespace_is_normalised(self) -> None:
        for section in parse_pmc_xml(SAMPLE)[0].sections:
            assert "  " not in section.text
            assert section.text == section.text.strip()

    def test_an_empty_articleset_yields_nothing(self) -> None:
        assert parse_pmc_xml("<pmc-articleset></pmc-articleset>") == []

    def test_invalid_xml_is_an_error(self) -> None:
        with pytest.raises(LiteratureSourceParseError, match="not valid XML"):
            parse_pmc_xml("<article><body>")


def _section(article: FullText, label: str) -> str:
    return next(s.text for s in article.sections if s.label == label)
