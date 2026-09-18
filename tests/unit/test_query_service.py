"""Tests for the local retrieval use case over a real in-memory store."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.adapters.embedding import EmbeddingError
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import embed_corpus, ingest_papers, query_passages
from bio_research_copilot.application.query import RetrieverFactory, build_dense
from bio_research_copilot.domain import Paper, SearchQuery, Source
from tests.unit.fakes import FakeEmbeddingModel

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


# -- dense retrieval -----------------------------------------------------------


def dense_factory(model: FakeEmbeddingModel) -> RetrieverFactory:
    return lambda repository: build_dense(repository, model)


def test_dense_requires_vectors_to_exist(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    with pytest.raises(EmbeddingError, match="6 of 6 passages have no vector"):
        query_passages(repo, "CRISPR", 3, build_retriever=dense_factory(model))


def test_dense_refuses_a_partially_embedded_corpus(repo: SqliteRepository) -> None:
    """Searching a partial corpus would silently understate recall, so it fails instead."""
    model = FakeEmbeddingModel()
    embed_corpus(repo, model)
    repo._conn.execute("DELETE FROM embeddings WHERE chunk_id = 'pubmed:24157548:1'")
    repo._conn.commit()

    with pytest.raises(EmbeddingError, match="1 of 6 passages have no vector"):
        query_passages(repo, "CRISPR", 3, build_retriever=dense_factory(model))


def test_dense_returns_ranked_evidence(repo: SqliteRepository) -> None:
    model = FakeEmbeddingModel()
    embed_corpus(repo, model)

    evidence = query_passages(repo, "CRISPR", 3, build_retriever=dense_factory(model))

    assert len(evidence) == 3
    assert [e.rank for e in evidence] == [1, 2, 3]
    assert all(e.method == "dense:fake-v1" for e in evidence)
    assert all("cosine" in e.components for e in evidence)
    scores = [e.score for e in evidence]
    assert scores == sorted(scores, reverse=True)


def test_dense_always_returns_results_unlike_bm25(repo: SqliteRepository) -> None:
    """Every vector has some cosine similarity, so dense retrieval never abstains."""
    model = FakeEmbeddingModel()
    embed_corpus(repo, model)

    assert query_passages(repo, "zebrafish husbandry", 5) == []
    dense = query_passages(repo, "zebrafish husbandry", 5, build_retriever=dense_factory(model))
    assert len(dense) == 5


def test_dense_on_an_empty_store_is_empty() -> None:
    empty = SqliteRepository.open(":memory:")
    assert (
        query_passages(empty, "CRISPR", 5, build_retriever=dense_factory(FakeEmbeddingModel()))
        == []
    )
