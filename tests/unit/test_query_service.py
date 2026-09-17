"""Tests for the local retrieval use case over a real in-memory store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import ingest_papers, query_passages
from bio_research_copilot.domain import Paper, SearchQuery, Source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


class FakeSource:
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers

    def search(self, query: SearchQuery) -> list[Paper]:
        return self.papers


@pytest.fixture
def repo() -> SqliteRepository:
    repo = SqliteRepository.open(":memory:")
    ingest_papers(FakeSource(RECORDED), repo, "q", 3, clock=lambda: T1)
    return repo


def test_all_passages_is_ordered_by_chunk_id(repo: SqliteRepository) -> None:
    ids = [p.chunk_id for p in repo.all_passages()]
    assert ids == sorted(ids)
    assert len(ids) == 6


def test_query_returns_ranked_evidence_from_the_store(repo: SqliteRepository) -> None:
    evidence = query_passages(repo, "double nicking strategy off-target cleavage", 3)
    assert evidence
    assert evidence[0].passage.source_id == "24157548"
    assert evidence[0].rank == 1
    assert evidence[0].method == "bm25-v1"
    assert all(e.passage.source is Source.PUBMED for e in evidence)


def test_query_honours_max_results(repo: SqliteRepository) -> None:
    assert len(query_passages(repo, "CRISPR Cas9", 2)) == 2


def test_query_with_no_overlap_is_empty(repo: SqliteRepository) -> None:
    assert query_passages(repo, "zebrafish husbandry", 5) == []


def test_empty_store_is_empty_result() -> None:
    assert query_passages(SqliteRepository.open(":memory:"), "CRISPR", 5) == []


def test_blank_question_is_rejected(repo: SqliteRepository) -> None:
    with pytest.raises(ValidationError):
        query_passages(repo, "   ", 5)
