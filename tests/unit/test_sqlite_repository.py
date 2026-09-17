"""Tests for the SQLite repository: idempotent upsert, provenance, and durability."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.persistence import PersistenceError, RetrievedPaperRef
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.domain import Paper, PublicationDate, Source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def repo() -> SqliteRepository:
    return SqliteRepository.open(":memory:")


def minimal(pmid: str = "1", **overrides: object) -> Paper:
    fields: dict[str, object] = {
        "source": Source.PUBMED,
        "source_id": pmid,
        "title": f"Paper {pmid}",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }
    fields.update(overrides)
    return Paper.model_validate(fields)


class TestUpsert:
    def test_round_trips_recorded_papers_losslessly(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        for original in RECORDED:
            assert repo.get_paper(Source.PUBMED, original.source_id) == original

    def test_first_insert_counts_as_inserted(self, repo: SqliteRepository) -> None:
        result = repo.upsert_papers(RECORDED, seen_at=T1)
        assert (result.inserted, result.updated) == (3, 0)
        assert repo.count_papers() == 3

    def test_reingest_is_idempotent(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        result = repo.upsert_papers(RECORDED, seen_at=T2)
        assert (result.inserted, result.updated) == (0, 3)
        assert repo.count_papers() == 3

    def test_reingest_refreshes_fields_but_keeps_first_seen(self, repo: SqliteRepository) -> None:
        repo.upsert_papers([minimal("9", title="Old title")], seen_at=T1)
        repo.upsert_papers([minimal("9", title="Corrected title")], seen_at=T2)
        stored = repo.get_paper(Source.PUBMED, "9")
        assert stored is not None
        assert stored.title == "Corrected title"
        row = repo._conn.execute(
            "SELECT first_seen_at, last_seen_at FROM papers WHERE source_id = '9'"
        ).fetchone()
        assert row["first_seen_at"] == T1.isoformat()
        assert row["last_seen_at"] == T2.isoformat()

    def test_optional_fields_survive_as_none(self, repo: SqliteRepository) -> None:
        repo.upsert_papers([minimal("5")], seen_at=T1)
        stored = repo.get_paper(Source.PUBMED, "5")
        assert stored is not None
        assert stored.abstract is None
        assert stored.journal is None
        assert stored.publication_date is None
        assert stored.authors == []

    def test_partial_date_survives(self, repo: SqliteRepository) -> None:
        paper = minimal("6", publication_date=PublicationDate(year=2019, month=3))
        repo.upsert_papers([paper], seen_at=T1)
        stored = repo.get_paper(Source.PUBMED, "6")
        assert stored is not None
        assert stored.publication_date == PublicationDate(year=2019, month=3)

    def test_unicode_survives(self, repo: SqliteRepository) -> None:
        paper = minimal("7", title="Amyloid-β and GSK3β", authors=["Müller J", "Ørsted H"])
        repo.upsert_papers([paper], seen_at=T1)
        assert repo.get_paper(Source.PUBMED, "7") == paper

    def test_missing_paper_is_none(self, repo: SqliteRepository) -> None:
        assert repo.get_paper(Source.PUBMED, "does-not-exist") is None

    def test_naive_timestamp_is_rejected(self, repo: SqliteRepository) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.upsert_papers([minimal()], seen_at=datetime(2026, 1, 1))


class TestRetrievals:
    def test_records_papers_in_rank_order(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        rid = repo.record_retrieval("CRISPR off-target", 3, RECORDED, retrieved_at=T1)
        record = repo.get_retrieval(rid)
        assert record is not None
        assert record.question == "CRISPR off-target"
        assert record.max_results == 3
        assert record.retrieved_at == T1
        assert record.papers == [
            RetrievedPaperRef(source=Source.PUBMED, source_id=p.source_id, rank=i)
            for i, p in enumerate(RECORDED, start=1)
        ]

    def test_each_run_gets_its_own_id(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        first = repo.record_retrieval("q", 3, RECORDED, retrieved_at=T1)
        second = repo.record_retrieval("q", 3, RECORDED, retrieved_at=T2)
        assert first != second

    def test_empty_retrieval_is_recorded(self, repo: SqliteRepository) -> None:
        rid = repo.record_retrieval("zzqxv", 5, [], retrieved_at=T1)
        record = repo.get_retrieval(rid)
        assert record is not None
        assert record.papers == []

    def test_unknown_paper_is_rejected(self, repo: SqliteRepository) -> None:
        with pytest.raises(PersistenceError, match="not stored"):
            repo.record_retrieval("q", 1, [minimal("404")], retrieved_at=T1)

    def test_missing_retrieval_is_none(self, repo: SqliteRepository) -> None:
        assert repo.get_retrieval(999) is None


class TestDurability:
    def test_data_survives_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "store.sqlite"
        with SqliteRepository.open(db) as first:
            first.upsert_papers(RECORDED, seen_at=T1)
        with SqliteRepository.open(db) as second:
            assert second.count_papers() == 3
            assert second.get_paper(Source.PUBMED, "24157548") == RECORDED[0]

    def test_reopening_does_not_recreate_schema(self, tmp_path: Path) -> None:
        db = tmp_path / "store.sqlite"
        with SqliteRepository.open(db) as first:
            first.upsert_papers([minimal()], seen_at=T1)
        with SqliteRepository.open(db) as second:
            version = second._conn.execute("PRAGMA user_version").fetchone()[0]
            assert version == 1
            assert second.count_papers() == 1

    def test_unopenable_path_is_persistence_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        with pytest.raises(PersistenceError, match="could not open"):
            SqliteRepository.open(blocker / "store.sqlite")
