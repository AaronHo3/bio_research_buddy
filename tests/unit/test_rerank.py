"""Tests for the reranking stage.

The fake reranker scores by an arbitrary rule, so these assert the wrapper's
mechanics: that it reorders by whatever the reranker returns, that it only ever
reorders, that it reads deeper than the cut-off, and that it keeps the first
stage's scores for tracing. Whether reranking actually helps is the benchmark's
question, not a unit test's.
"""

import hashlib

import pytest

from bio_research_copilot.adapters.reranking import RerankerError
from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery, Source
from bio_research_copilot.retrieval import RerankingRetriever
from tests.unit.fakes import FakeReranker, failing_reranker


def passage(chunk: str, text: str) -> Passage:
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


class StubRetriever:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.name = "stub-v1"
        self.items = items
        self.depths: list[int] = []

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        self.depths.append(query.max_results)
        return [
            RetrievedEvidence(
                passage=passage(chunk, text),
                rank=rank,
                score=100.0 - rank,
                components={"stub-v1": 100.0 - rank},
                method=self.name,
            )
            for rank, (chunk, text) in enumerate(self.items[: query.max_results], start=1)
        ]


SHORT_FIRST = [("1:0", "a"), ("2:0", "bb"), ("3:0", "ccc")]


def ids(results: list[RetrievedEvidence]) -> list[str]:
    return [r.passage.chunk_id for r in results]


def test_reorders_by_the_reranker_score() -> None:
    """The stub ranks shortest first; the reranker scores by length, so it inverts."""
    base = StubRetriever(SHORT_FIRST)
    results = RerankingRetriever(base, FakeReranker()).search(SearchQuery(raw="q"))
    assert ids(results) == ["pubmed:3:0", "pubmed:2:0", "pubmed:1:0"]
    assert [r.rank for r in results] == [1, 2, 3]
    assert [r.score for r in results] == [3.0, 2.0, 1.0]


def test_only_reorders_and_never_adds_or_drops() -> None:
    base = StubRetriever(SHORT_FIRST)
    before = {r.passage.chunk_id for r in base.search(SearchQuery(raw="q", max_results=25))}
    after = set(ids(RerankingRetriever(base, FakeReranker()).search(SearchQuery(raw="q"))))
    assert after == before


def test_keeps_the_first_stage_scores_for_tracing() -> None:
    base = StubRetriever(SHORT_FIRST)
    top = RerankingRetriever(base, FakeReranker()).search(SearchQuery(raw="q"))[0]
    assert top.components == {"stub-v1": pytest.approx(97.0), "rerank:fake-rerank-v1": 3.0}
    assert top.method == "rerank:fake-rerank-v1(stub-v1)"


def test_reads_deeper_than_the_cut_off() -> None:
    """A relevant passage outside the shortlist can never be recovered by reranking."""
    base = StubRetriever([(f"{i}:0", "x" * i) for i in range(1, 40)])
    results = RerankingRetriever(base, FakeReranker(), pool=25).search(
        SearchQuery(raw="q", max_results=3)
    )
    assert base.depths == [25]
    assert len(results) == 3
    # The deepest candidate in the pool has the longest text and so wins.
    assert ids(results)[0] == "pubmed:25:0"


def test_a_large_cut_off_wins_over_the_pool() -> None:
    base = StubRetriever([(f"{i}:0", "x") for i in range(1, 90)])
    RerankingRetriever(base, FakeReranker(), pool=10).search(SearchQuery(raw="q", max_results=40))
    assert base.depths == [40]


def test_depth_is_capped_at_the_query_limit() -> None:
    base = StubRetriever([(f"{i}:0", "x") for i in range(1, 200)])
    RerankingRetriever(base, FakeReranker(), pool=500).search(SearchQuery(raw="q"))
    assert base.depths == [100]


def test_the_query_and_texts_reach_the_reranker() -> None:
    base = StubRetriever(SHORT_FIRST)
    reranker = FakeReranker()
    RerankingRetriever(base, reranker).search(SearchQuery(raw="what causes it?"))
    assert reranker.calls == [("what causes it?", ["a", "bb", "ccc"])]


def test_ties_break_by_chunk_id() -> None:
    base = StubRetriever([("9:0", "same"), ("2:0", "same")])
    results = RerankingRetriever(base, FakeReranker()).search(SearchQuery(raw="q"))
    assert ids(results) == ["pubmed:2:0", "pubmed:9:0"]


def test_empty_shortlist_skips_the_reranker() -> None:
    reranker = FakeReranker()
    assert RerankingRetriever(StubRetriever([]), reranker).search(SearchQuery(raw="q")) == []
    assert reranker.calls == []


def test_reranker_failure_propagates() -> None:
    base = StubRetriever(SHORT_FIRST)
    with pytest.raises(RerankerError, match="reranker unavailable"):
        RerankingRetriever(base, failing_reranker()).search(SearchQuery(raw="q"))


def test_pool_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pool must be at least 1"):
        RerankingRetriever(StubRetriever([]), FakeReranker(), pool=0)


def test_name_records_both_stages() -> None:
    wrapper = RerankingRetriever(StubRetriever([]), FakeReranker(name="ce-v2"))
    assert wrapper.name == "rerank:ce-v2(stub-v1)"
