"""Tests for the embedding use case: incremental work, batching, and failure handling."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.embedding import EmbeddingError
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import embed_corpus, load_corpus
from bio_research_copilot.domain import Paper, SectionChunker, Source
from tests.unit.fakes import FakeEmbeddingModel, failing_model

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
def repo() -> SqliteRepository:
    repo = SqliteRepository.open(":memory:")
    load_corpus(repo, RECORDED, label="fixture", clock=lambda: T1)
    return repo


def test_first_run_embeds_every_passage(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel(dimension=8)
    report = embed_corpus(repo, model)

    assert report.model_name == "fake-v1"
    assert report.dimension == 8
    assert report.passages_total == 6
    assert report.passages_embedded == 6
    assert report.passages_already_current == 0
    assert report.embeddings_stored == 6
    assert repo.count_embeddings("fake-v1") == 6


def test_second_run_embeds_nothing(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    embed_corpus(repo, model)
    model.batches.clear()

    report = embed_corpus(repo, model)
    assert report.passages_embedded == 0
    assert report.passages_already_current == 6
    assert report.dimension is None
    assert model.batches == []


def test_changed_text_is_re_embedded_and_nothing_else_is(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    embed_corpus(repo, model)
    before = {e.chunk_id: e.vector for e in repo.get_embeddings("fake-v1")}

    edited = Paper(
        source=Source.PUBMED,
        source_id=RECORDED[0].source_id,
        title=RECORDED[0].title,
        abstract="A completely rewritten abstract.",
        url=RECORDED[0].url,
    )
    repo.upsert_papers([edited], seen_at=T1)
    repo.replace_passages(SectionChunker().split(edited))

    model.batches.clear()
    report = embed_corpus(repo, model)

    assert report.passages_embedded == 1
    assert model.batches == [["A completely rewritten abstract."]]
    after = {e.chunk_id: e.vector for e in repo.get_embeddings("fake-v1")}
    changed = [cid for cid in before if before[cid] != after.get(cid)]
    assert changed == [f"pubmed:{RECORDED[0].source_id}:1"]


def test_a_different_model_embeds_from_scratch(repo: SqliteRepository) -> None:
    embed_corpus(repo, FakeEmbeddingModel(name="a"))
    report = embed_corpus(repo, FakeEmbeddingModel(name="b"))

    assert report.passages_embedded == 6
    assert repo.count_embeddings("a") == 6
    assert repo.count_embeddings("b") == 6


def test_batches_respect_the_batch_size(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    embed_corpus(repo, model, batch_size=4)
    assert [len(b) for b in model.batches] == [4, 2]


def test_batch_size_must_be_positive(repo: SqliteRepository) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        embed_corpus(repo, FakeEmbeddingModel(), batch_size=0)


def test_empty_store_is_a_no_op() -> None:
    report = embed_corpus(SqliteRepository.open(":memory:"), FakeEmbeddingModel())
    assert report.passages_total == 0
    assert report.passages_embedded == 0
    assert report.dimension is None


def test_model_failure_propagates_and_stores_nothing(repo: SqliteRepository) -> None:
    with pytest.raises(EmbeddingError, match="model unavailable"):
        embed_corpus(repo, failing_model())
    assert repo.count_embeddings("fake-v1") == 0


def test_wrong_vector_count_is_an_error(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    model.wrong_count = True
    with pytest.raises(EmbeddingError, match="returned 5 vectors for 6 passages"):
        embed_corpus(repo, model)


def test_a_failed_batch_keeps_earlier_batches(repo: SqliteRepository) -> None:
    """Batches commit independently, so a late failure leaves partial progress."""

    class FailsOnSecondBatch(FakeEmbeddingModel):
        def embed_passages(self, texts: list[str]) -> list[list[float]]:
            if self.batches:
                msg = "died mid-run"
                raise EmbeddingError(msg)
            return super().embed_passages(texts)

    with pytest.raises(EmbeddingError, match="died mid-run"):
        embed_corpus(repo, FailsOnSecondBatch(), batch_size=4)
    assert repo.count_embeddings("fake-v1") == 4
    # Re-running finishes the job rather than starting over.
    assert embed_corpus(repo, FakeEmbeddingModel(), batch_size=4).passages_embedded == 2
