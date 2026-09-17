"""Tests for the search use case and the import boundary it depends on."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.application import search_papers
from bio_research_copilot.domain import Paper, SearchQuery, Source

SRC = Path(__file__).parent.parent.parent / "src" / "bio_research_copilot"


class FakeSource:
    """A LiteratureSource that records what it was asked and returns canned papers."""

    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery) -> list[Paper]:
        self.queries.append(query)
        return self.papers


def paper(pmid: str) -> Paper:
    return Paper(
        source=Source.PUBMED,
        source_id=pmid,
        title=f"Paper {pmid}",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


def test_builds_query_and_delegates_to_source() -> None:
    source = FakeSource([paper("1"), paper("2")])
    result = search_papers(source, "  tau phosphorylation ", max_results=2)
    assert result == source.papers
    assert source.queries == [SearchQuery(raw="tau phosphorylation", max_results=2)]


def test_blank_question_is_rejected_before_reaching_source() -> None:
    source = FakeSource([])
    with pytest.raises(ValidationError):
        search_papers(source, "   ", max_results=5)
    assert source.queries == []


def test_limit_out_of_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        search_papers(FakeSource([]), "q", max_results=0)


@pytest.mark.parametrize("layer", ["domain", "application"])
def test_inner_layers_do_not_import_the_pubmed_adapter(layer: str) -> None:
    """The port is the only adapter symbol inner layers may see."""
    for path in (SRC / layer).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "adapters.pubmed" not in text, f"{path} imports the concrete PubMed adapter"
        assert "httpx" not in text, f"{path} imports an HTTP library"


def test_domain_does_not_import_adapters_at_all() -> None:
    for path in (SRC / "domain").rglob("*.py"):
        assert "adapters" not in path.read_text(encoding="utf-8"), path
