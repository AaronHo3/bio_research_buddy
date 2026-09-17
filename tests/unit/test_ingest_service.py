"""Tests for the ingestion use case with a fake source and a real in-memory store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import ingest_papers
from bio_research_copilot.domain import Paper, SearchQuery, Source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


class FakeSource:
    def __init__(self, papers: list[Paper]) -> None:
        self.papers = papers
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery) -> list[Paper]:
        self.queries.append(query)
        return self.papers


@pytest.fixture
def repo() -> SqliteRepository:
    return SqliteRepository.open(":memory:")


def test_first_ingest_stores_papers_and_provenance(repo: SqliteRepository) -> None:
    source = FakeSource(RECORDED)
    report = ingest_papers(source, repo, "CRISPR off-target", 3, clock=lambda: T1)

    assert source.queries == [SearchQuery(raw="CRISPR off-target", max_results=3)]
    assert report.papers_fetched == 3
    assert report.papers_inserted == 3
    assert report.papers_updated == 0
    assert report.paper_ids == ["24157548", "28375731", "25315507"]
    assert report.retrieved_at == T1
    assert repo.count_papers() == 3

    record = repo.get_retrieval(report.retrieval_id)
    assert record is not None
    assert [p.source_id for p in record.papers] == report.paper_ids


def test_second_ingest_adds_nothing_but_records_a_new_run(repo: SqliteRepository) -> None:
    source = FakeSource(RECORDED)
    first = ingest_papers(source, repo, "q", 3, clock=lambda: T1)
    second = ingest_papers(source, repo, "q", 3, clock=lambda: T1)

    assert second.papers_inserted == 0
    assert second.papers_updated == 3
    assert repo.count_papers() == 3
    assert second.retrieval_id != first.retrieval_id


def test_empty_result_still_records_the_run(repo: SqliteRepository) -> None:
    report = ingest_papers(FakeSource([]), repo, "zzqxv", 5, clock=lambda: T1)
    assert report.papers_fetched == 0
    assert repo.get_retrieval(report.retrieval_id) is not None


def test_invalid_question_never_reaches_source_or_store(repo: SqliteRepository) -> None:
    source = FakeSource(RECORDED)
    with pytest.raises(ValidationError):
        ingest_papers(source, repo, "   ", 3)
    assert source.queries == []
    assert repo.count_papers() == 0


def test_ingest_writes_passages_and_reports_strategy(repo: SqliteRepository) -> None:
    report = ingest_papers(FakeSource(RECORDED), repo, "q", 3, clock=lambda: T1)
    assert report.passages_written == 6  # title + abstract for each of 3 papers
    assert report.chunking_strategy == "section-v1"
    assert repo.count_passages() == 6
    stored = repo.get_passages(Source.PUBMED, "24157548")
    assert [p.section for p in stored] == ["title", "abstract"]
    assert stored[1].text == RECORDED[0].abstract


def test_reingest_does_not_duplicate_passages(repo: SqliteRepository) -> None:
    source = FakeSource(RECORDED)
    ingest_papers(source, repo, "q", 3, clock=lambda: T1)
    ingest_papers(source, repo, "q", 3, clock=lambda: T1)
    assert repo.count_passages() == 6


def test_stored_paper_matches_fetched_paper(repo: SqliteRepository) -> None:
    ingest_papers(FakeSource(RECORDED), repo, "q", 3, clock=lambda: T1)
    assert repo.get_paper(Source.PUBMED, "28375731") == RECORDED[1]
