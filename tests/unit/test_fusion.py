"""Tests for reciprocal rank fusion.

Every expectation is computed by hand from the formula, so a failure names the
part that broke rather than just "the ranking changed".
"""

import hashlib

import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery, Source
from bio_research_copilot.retrieval import ReciprocalRankFusion

CONSTANT = 60.0


def passage(chunk: str) -> Passage:
    pmid, position = chunk.split(":")
    text = f"text {chunk}"
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


class StubRetriever:
    """Returns a fixed ranking and records how deep it was asked to go."""

    def __init__(self, name: str, chunks: list[str]) -> None:
        self.name = name
        self.chunks = chunks
        self.depths: list[int] = []

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        self.depths.append(query.max_results)
        return [
            RetrievedEvidence(
                passage=passage(chunk),
                rank=rank,
                score=1.0 / rank,
                components={self.name: 1.0 / rank},
                method=self.name,
            )
            for rank, chunk in enumerate(self.chunks[: query.max_results], start=1)
        ]


def rrf(rank: int, constant: float = CONSTANT) -> float:
    return 1.0 / (constant + rank)


def ids(results: list[RetrievedEvidence]) -> list[str]:
    return [r.passage.chunk_id for r in results]


def test_agreement_beats_any_single_first_place() -> None:
    """A passage both retrievers rank second beats one that only one ranks first."""
    a = StubRetriever("a", ["1:0", "2:0"])
    b = StubRetriever("b", ["3:0", "2:0"])
    results = ReciprocalRankFusion([a, b]).search(SearchQuery(raw="q"))

    assert ids(results)[0] == "pubmed:2:0"
    assert results[0].score == pytest.approx(rrf(2) + rrf(2))
    assert results[1].score == pytest.approx(rrf(1))


def test_components_name_each_retriever_and_sum_to_the_score() -> None:
    a = StubRetriever("a", ["1:0"])
    b = StubRetriever("b", ["2:0", "1:0"])
    top = ReciprocalRankFusion([a, b]).search(SearchQuery(raw="q"))
    shared = next(r for r in top if r.passage.chunk_id == "pubmed:1:0")

    assert shared.components == {"a": pytest.approx(rrf(1)), "b": pytest.approx(rrf(2))}
    assert shared.score == pytest.approx(sum(shared.components.values()))


def test_a_passage_only_one_retriever_found_still_appears() -> None:
    a = StubRetriever("a", ["1:0"])
    b = StubRetriever("b", ["2:0"])
    results = ReciprocalRankFusion([a, b]).search(SearchQuery(raw="q"))

    assert set(ids(results)) == {"pubmed:1:0", "pubmed:2:0"}
    assert all(len(r.components) == 1 for r in results)


def test_one_retriever_preserves_its_order() -> None:
    only = StubRetriever("a", ["3:0", "1:0", "2:0"])
    assert ids(ReciprocalRankFusion([only]).search(SearchQuery(raw="q"))) == [
        "pubmed:3:0",
        "pubmed:1:0",
        "pubmed:2:0",
    ]


def test_reads_deeper_than_the_requested_cut_off() -> None:
    """Fusing only the final k would discard the agreement evidence."""
    a = StubRetriever("a", [f"{i}:0" for i in range(1, 60)])
    fusion = ReciprocalRankFusion([a], pool=50)
    results = fusion.search(SearchQuery(raw="q", max_results=3))

    assert a.depths == [50]
    assert len(results) == 3


def test_a_large_cut_off_wins_over_the_pool() -> None:
    a = StubRetriever("a", [f"{i}:0" for i in range(1, 90)])
    ReciprocalRankFusion([a], pool=10).search(SearchQuery(raw="q", max_results=80))
    assert a.depths == [80]


def test_depth_is_capped_at_the_query_limit() -> None:
    a = StubRetriever("a", [f"{i}:0" for i in range(1, 200)])
    ReciprocalRankFusion([a], pool=500).search(SearchQuery(raw="q", max_results=10))
    assert a.depths == [100]


def test_ranks_start_at_one_and_respect_max_results() -> None:
    a = StubRetriever("a", ["1:0", "2:0", "3:0"])
    results = ReciprocalRankFusion([a]).search(SearchQuery(raw="q", max_results=2))
    assert [r.rank for r in results] == [1, 2]


def test_ties_break_by_chunk_id() -> None:
    a = StubRetriever("a", ["9:0"])
    b = StubRetriever("b", ["2:0"])
    results = ReciprocalRankFusion([a, b]).search(SearchQuery(raw="q"))
    assert ids(results) == ["pubmed:2:0", "pubmed:9:0"]
    assert results[0].score == pytest.approx(results[1].score)


def test_a_larger_constant_flattens_the_top() -> None:
    """With a big constant, ranks matter less and agreement matters more."""
    a = StubRetriever("a", ["1:0", "2:0"])
    b = StubRetriever("b", ["2:0", "1:0"])
    flat = ReciprocalRankFusion([a, b], constant=1000.0).search(SearchQuery(raw="q"))
    assert flat[0].score == pytest.approx(flat[1].score)


def test_weights_shift_the_balance() -> None:
    a = StubRetriever("a", ["1:0"])
    b = StubRetriever("b", ["2:0"])
    results = ReciprocalRankFusion([a, b], weights=[1.0, 5.0]).search(SearchQuery(raw="q"))
    assert ids(results) == ["pubmed:2:0", "pubmed:1:0"]
    assert results[0].score == pytest.approx(5.0 * rrf(1))


def test_name_records_what_was_fused() -> None:
    fusion = ReciprocalRankFusion([StubRetriever("bm25-v1", []), StubRetriever("dense:x", [])])
    assert fusion.name == "rrf(bm25-v1+dense:x)"


def test_empty_rankings_give_an_empty_result() -> None:
    fusion = ReciprocalRankFusion([StubRetriever("a", []), StubRetriever("b", [])])
    assert fusion.search(SearchQuery(raw="q")) == []


def test_fusion_is_order_independent() -> None:
    chunks_a, chunks_b = ["1:0", "2:0", "3:0"], ["3:0", "1:0"]
    forward = ReciprocalRankFusion(
        [StubRetriever("a", chunks_a), StubRetriever("b", chunks_b)]
    ).search(SearchQuery(raw="q"))
    backward = ReciprocalRankFusion(
        [StubRetriever("b", chunks_b), StubRetriever("a", chunks_a)]
    ).search(SearchQuery(raw="q"))
    assert ids(forward) == ids(backward)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"constant": 0.0}, "constant must be positive"),
        ({"constant": -1.0}, "constant must be positive"),
        ({"pool": 0}, "pool must be at least 1"),
        ({"weights": [1.0]}, "got 1 weights for 2 retrievers"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    retrievers = [StubRetriever("a", []), StubRetriever("b", [])]
    with pytest.raises(ValueError, match=message):
        ReciprocalRankFusion(retrievers, **kwargs)  # type: ignore[arg-type]


def test_no_retrievers_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        ReciprocalRankFusion([])
