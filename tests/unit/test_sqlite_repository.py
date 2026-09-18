"""Tests for the SQLite repository: idempotent upsert, provenance, and durability."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.persistence import PersistenceError, RetrievedPaperRef
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.domain import (
    Paper,
    Passage,
    PassageEmbedding,
    PublicationDate,
    SectionChunker,
    Source,
)

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


class TestPassages:
    def test_round_trips_in_position_order(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        expected = SectionChunker().split(RECORDED[0])
        assert repo.replace_passages(list(reversed(expected))) == 2
        assert repo.get_passages(Source.PUBMED, RECORDED[0].source_id) == expected

    def test_replace_removes_stale_chunks(self, repo: SqliteRepository) -> None:
        long_paper = minimal("8", abstract="A: one.\n\nB: two.\n\nC: three.")
        repo.upsert_papers([long_paper], seen_at=T1)
        repo.replace_passages(SectionChunker().split(long_paper))
        assert repo.count_passages() == 4

        short_paper = minimal("8", abstract="Only one now.")
        repo.upsert_papers([short_paper], seen_at=T2)
        repo.replace_passages(SectionChunker().split(short_paper))
        assert [p.text for p in repo.get_passages(Source.PUBMED, "8")] == [
            "Paper 8",
            "Only one now.",
        ]

    def test_replace_touches_only_the_papers_given(self, repo: SqliteRepository) -> None:
        repo.upsert_papers(RECORDED, seen_at=T1)
        for paper in RECORDED:
            repo.replace_passages(SectionChunker().split(paper))
        repo.replace_passages(SectionChunker().split(RECORDED[1]))
        assert repo.count_passages() == 6

    def test_unknown_paper_is_rejected(self, repo: SqliteRepository) -> None:
        with pytest.raises(PersistenceError, match="not stored"):
            repo.replace_passages(SectionChunker().split(minimal("404")))

    def test_unknown_paper_has_no_passages(self, repo: SqliteRepository) -> None:
        assert repo.get_passages(Source.PUBMED, "nope") == []

    def test_empty_list_is_a_no_op(self, repo: SqliteRepository) -> None:
        assert repo.replace_passages([]) == 0


class TestEmbeddings:
    def embedding(self, chunk_id: str, content_hash: str, model: str = "m") -> PassageEmbedding:
        return PassageEmbedding(
            chunk_id=chunk_id, model=model, content_hash=content_hash, vector=[0.5, -0.25, 0.125]
        )

    def stored(self, repo: SqliteRepository) -> list[Passage]:
        repo.upsert_papers(RECORDED, seen_at=T1)
        passages = SectionChunker().split(RECORDED[0])
        repo.replace_passages(passages)
        return passages

    def test_round_trips_vectors(self, repo: SqliteRepository) -> None:
        p = self.stored(repo)[0]
        e = self.embedding(p.chunk_id, p.content_hash)
        assert repo.upsert_embeddings([e]) == 1
        assert repo.get_embeddings("m") == [e]
        assert repo.get_embeddings("m")[0].dimension == 3

    def test_float32_storage_keeps_enough_precision(self, repo: SqliteRepository) -> None:
        p = self.stored(repo)[0]
        vector = [0.1234567, -0.9876543, 1e-7]
        repo.upsert_embeddings(
            [
                PassageEmbedding(
                    chunk_id=p.chunk_id, model="m", content_hash=p.content_hash, vector=vector
                )
            ]
        )
        back = repo.get_embeddings("m")[0].vector
        assert back == pytest.approx(vector, rel=1e-6, abs=1e-9)

    def test_upsert_replaces_the_same_chunk_and_model(self, repo: SqliteRepository) -> None:
        p = self.stored(repo)[0]
        repo.upsert_embeddings([self.embedding(p.chunk_id, p.content_hash)])
        updated = PassageEmbedding(
            chunk_id=p.chunk_id, model="m", content_hash=p.content_hash, vector=[1.0, 1.0, 1.0]
        )
        repo.upsert_embeddings([updated])
        assert repo.get_embeddings("m") == [updated]
        assert repo.count_embeddings("m") == 1

    def test_models_are_stored_side_by_side(self, repo: SqliteRepository) -> None:
        p = self.stored(repo)[0]
        repo.upsert_embeddings(
            [
                self.embedding(p.chunk_id, p.content_hash, "a"),
                self.embedding(p.chunk_id, p.content_hash, "b"),
            ]
        )
        assert repo.count_embeddings("a") == 1
        assert repo.count_embeddings("b") == 1
        assert repo.get_embeddings("unknown") == []

    def test_unknown_passage_is_rejected(self, repo: SqliteRepository) -> None:
        with pytest.raises(PersistenceError, match="not stored"):
            repo.upsert_embeddings([self.embedding("pubmed:nope:0", "f" * 64)])

    def test_pending_lists_everything_when_nothing_is_embedded(
        self, repo: SqliteRepository
    ) -> None:
        passages = self.stored(repo)
        assert repo.passages_needing_embedding("m") == passages

    def test_pending_excludes_current_vectors(self, repo: SqliteRepository) -> None:
        passages = self.stored(repo)
        repo.upsert_embeddings([self.embedding(passages[0].chunk_id, passages[0].content_hash)])
        assert repo.passages_needing_embedding("m") == passages[1:]

    def test_pending_includes_stale_vectors(self, repo: SqliteRepository) -> None:
        passages = self.stored(repo)
        repo.upsert_embeddings([self.embedding(passages[0].chunk_id, "0" * 64)])
        assert repo.passages_needing_embedding("m") == passages

    def test_pending_is_per_model(self, repo: SqliteRepository) -> None:
        passages = self.stored(repo)
        repo.upsert_embeddings(
            [self.embedding(passages[0].chunk_id, passages[0].content_hash, "a")]
        )
        assert len(repo.passages_needing_embedding("a")) == len(passages) - 1
        assert len(repo.passages_needing_embedding("b")) == len(passages)

    def test_empty_upsert_is_a_no_op(self, repo: SqliteRepository) -> None:
        assert repo.upsert_embeddings([]) == 0

    def test_replacing_passages_discards_their_vectors(self, repo: SqliteRepository) -> None:
        """A vector for a passage that no longer exists is meaningless, so it cascades away."""
        paper = minimal("8", abstract="A: one.\n\nB: two.\n\nC: three.")
        repo.upsert_papers([paper], seen_at=T1)
        passages = SectionChunker().split(paper)
        repo.replace_passages(passages)
        repo.upsert_embeddings([self.embedding(p.chunk_id, p.content_hash) for p in passages])
        assert repo.count_embeddings("m") == 4

        shorter = minimal("8", abstract="Only one now.")
        repo.upsert_papers([shorter], seen_at=T2)
        repo.replace_passages(SectionChunker().split(shorter))

        assert repo.count_passages() == 2
        # Chunks 2 and 3 are gone, so their vectors cascaded away. Chunks 0 and 1
        # survived, so their vectors did too, and chunk 1's is now stale.
        assert [e.chunk_id for e in repo.get_embeddings("m")] == ["pubmed:8:0", "pubmed:8:1"]
        assert [p.chunk_id for p in repo.passages_needing_embedding("m")] == ["pubmed:8:1"]

    def test_a_reinserted_passage_with_changed_text_needs_re_embedding(
        self, repo: SqliteRepository
    ) -> None:
        paper = minimal("8", abstract="Original.")
        repo.upsert_papers([paper], seen_at=T1)
        passages = SectionChunker().split(paper)
        repo.replace_passages(passages)
        repo.upsert_embeddings([self.embedding(p.chunk_id, p.content_hash) for p in passages])
        assert repo.passages_needing_embedding("m") == []

        edited = minimal("8", abstract="Rewritten.")
        repo.upsert_papers([edited], seen_at=T2)
        repo.replace_passages(SectionChunker().split(edited))
        pending = repo.passages_needing_embedding("m")
        assert [p.chunk_id for p in pending] == ["pubmed:8:1"]


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
            assert version == 3
            assert second.count_papers() == 1

    def test_unopenable_path_is_persistence_error(self, tmp_path: Path) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        with pytest.raises(PersistenceError, match="could not open"):
            SqliteRepository.open(blocker / "store.sqlite")
