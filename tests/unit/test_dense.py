"""Tests for dense retrieval.

The fake model's vectors are hashes, so they say nothing about meaning. These
tests therefore assert the mechanics only: that cosine similarity is computed
correctly, that ranking and tie-breaking are deterministic, and that dimension
mismatches fail loudly. Whether retrieval is any *good* is what the benchmark
measures, not a unit test.
"""

import hashlib
import math

import pytest

from bio_research_copilot.adapters.embedding import EmbeddingError
from bio_research_copilot.domain import Passage, SearchQuery, Source
from bio_research_copilot.retrieval import DenseIndex
from tests.unit.fakes import FakeEmbeddingModel


def passage(chunk: str, text: str = "text") -> Passage:
    pmid, position = chunk.split(":")
    return Passage(
        source=Source.PUBMED,
        source_id=pmid,
        chunk_id=f"pubmed:{chunk}",
        position=int(position),
        section="abstract",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        strategy="test",
    )


class FixedModel:
    """Returns a vector chosen by the caller, so similarities are hand-computable."""

    name = "fixed-v1"

    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("the dense index must not re-embed stored passages")

    def embed_query(self, text: str) -> list[float]:
        return self.query_vector


def test_ranks_by_cosine_similarity() -> None:
    entries = [
        (passage("1:0"), [1.0, 0.0]),  # identical direction to the query
        (passage("2:0"), [0.0, 1.0]),  # orthogonal
        (passage("3:0"), [-1.0, 0.0]),  # opposite
    ]
    results = DenseIndex(entries, FixedModel([1.0, 0.0])).search(SearchQuery(raw="q"))
    assert [r.passage.chunk_id for r in results] == ["pubmed:1:0", "pubmed:2:0", "pubmed:3:0"]
    assert [round(r.score, 6) for r in results] == [1.0, 0.0, -1.0]


def test_magnitude_does_not_affect_score() -> None:
    """Cosine compares direction only, so a longer vector is not a better match."""
    entries = [(passage("1:0"), [3.0, 0.0]), (passage("2:0"), [0.001, 0.0])]
    results = DenseIndex(entries, FixedModel([2.0, 0.0])).search(SearchQuery(raw="q"))
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0)


def test_hand_computed_similarity() -> None:
    entries = [(passage("1:0"), [1.0, 1.0])]
    result = DenseIndex(entries, FixedModel([1.0, 0.0])).search(SearchQuery(raw="q"))[0]
    assert result.score == pytest.approx(1 / math.sqrt(2))


def test_ranks_start_at_one_and_respect_max_results() -> None:
    entries = [(passage(f"{i}:0"), [float(i), 1.0]) for i in range(1, 6)]
    results = DenseIndex(entries, FixedModel([1.0, 0.0])).search(
        SearchQuery(raw="q", max_results=2)
    )
    assert [r.rank for r in results] == [1, 2]


def test_ties_break_by_chunk_id() -> None:
    entries = [(passage("9:0"), [1.0, 0.0]), (passage("2:0"), [1.0, 0.0])]
    results = DenseIndex(entries, FixedModel([1.0, 0.0])).search(SearchQuery(raw="q"))
    assert [r.passage.chunk_id for r in results] == ["pubmed:2:0", "pubmed:9:0"]
    assert results[0].score == results[1].score


def test_returns_every_passage_even_when_unrelated() -> None:
    """Unlike BM25, dense retrieval has no notion of 'no match'. It always ranks."""
    entries = [(passage(f"{i}:0"), [0.0, 1.0]) for i in range(1, 4)]
    results = DenseIndex(entries, FixedModel([1.0, 0.0])).search(SearchQuery(raw="q"))
    assert len(results) == 3
    assert all(r.score == pytest.approx(0.0) for r in results)


def test_empty_index_returns_nothing() -> None:
    assert DenseIndex([], FakeEmbeddingModel()).search(SearchQuery(raw="q")) == []


def test_evidence_carries_method_and_component() -> None:
    result = DenseIndex([(passage("1:0"), [1.0, 0.0])], FixedModel([1.0, 0.0])).search(
        SearchQuery(raw="q")
    )[0]
    assert result.method == "dense:fixed-v1"
    assert result.components == {"cosine": result.score}


def test_index_length_and_construction_order_do_not_matter() -> None:
    entries = [(passage("1:0"), [1.0, 0.0]), (passage("2:0"), [0.0, 1.0])]
    forward = DenseIndex(entries, FixedModel([1.0, 1.0])).search(SearchQuery(raw="q"))
    backward = DenseIndex(list(reversed(entries)), FixedModel([1.0, 1.0])).search(
        SearchQuery(raw="q")
    )
    assert forward == backward
    assert len(DenseIndex(entries, FixedModel([1.0, 0.0]))) == 2


def test_mismatched_stored_dimensions_are_rejected() -> None:
    entries = [(passage("1:0"), [1.0, 0.0]), (passage("2:0"), [1.0, 0.0, 0.0])]
    with pytest.raises(EmbeddingError, match="differing sizes"):
        DenseIndex(entries, FixedModel([1.0, 0.0]))


def test_query_of_the_wrong_dimension_is_rejected() -> None:
    index = DenseIndex([(passage("1:0"), [1.0, 0.0])], FixedModel([1.0, 0.0, 0.0]))
    with pytest.raises(EmbeddingError, match="re-run embed"):
        index.search(SearchQuery(raw="q"))


def test_all_zero_query_vector_is_rejected() -> None:
    index = DenseIndex([(passage("1:0"), [1.0, 0.0])], FixedModel([0.0, 0.0]))
    with pytest.raises(EmbeddingError, match="all-zero"):
        index.search(SearchQuery(raw="q"))


def test_all_zero_stored_vector_scores_zero_rather_than_failing() -> None:
    entries = [(passage("1:0"), [0.0, 0.0]), (passage("2:0"), [1.0, 0.0])]
    results = DenseIndex(entries, FixedModel([1.0, 0.0])).search(SearchQuery(raw="q"))
    assert results[0].passage.chunk_id == "pubmed:2:0"
    assert results[1].score == pytest.approx(0.0)
