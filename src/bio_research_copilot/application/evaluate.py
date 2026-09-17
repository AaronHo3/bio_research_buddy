"""Run a retriever over a benchmark and score it.

Judgments are at paper level, so the passage ranking is collapsed to a paper
ranking: each paper takes the rank of its best passage. More passages than
``k`` are requested so that ``k`` distinct papers can usually be filled.
"""

import time
from collections.abc import Callable
from datetime import datetime
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.application.ingest import utc_now
from bio_research_copilot.application.query import RetrieverFactory, build_bm25
from bio_research_copilot.domain import RetrievedEvidence, SearchQuery
from bio_research_copilot.evaluation import Benchmark, RankMetrics, rank_metrics

PASSAGES_PER_PAPER_GUESS = 4


def collapse_to_papers(evidence: list[RetrievedEvidence], k: int) -> list[str]:
    """Paper ids in order of their best passage, at most ``k``."""
    seen: list[str] = []
    for item in evidence:
        pid = item.passage.source_id
        if pid not in seen:
            seen.append(pid)
        if len(seen) == k:
            break
    return seen


class QuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    topic: str
    type: str
    question: str
    judged: bool
    ranked_papers: list[str]
    relevant: list[str]
    metrics: RankMetrics | None
    top_score: float | None
    latency_ms: float


class TypeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: int
    precision: float
    recall: float
    mrr: float
    ndcg: float


class UnanswerableResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    papers_returned: int
    top_score: float | None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    retriever: str
    k: int
    corpus_papers: int
    corpus_passages: int
    generated_at: datetime
    judged_questions: int
    unjudged_questions: list[str] = Field(description="Answerable ids with no judgments yet.")
    mean_precision: float | None
    mean_recall: float | None
    mean_mrr: float | None
    mean_ndcg: float | None
    mean_latency_ms: float
    by_type: dict[str, TypeSummary]
    unanswerable: list[UnanswerableResult]
    results: list[QuestionResult]


def evaluate_retrieval(
    repository: PaperRepository,
    benchmark: Benchmark,
    *,
    k: int = 10,
    build_retriever: RetrieverFactory = build_bm25,
    clock: Callable[[], datetime] = utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> EvaluationReport:
    passages = repository.all_passages()
    retriever = build_retriever(passages)
    results: list[QuestionResult] = []

    for q in benchmark.questions:
        query = SearchQuery(raw=q.question, max_results=min(100, k * PASSAGES_PER_PAPER_GUESS))
        started = timer()
        evidence = retriever.search(query)
        latency_ms = (timer() - started) * 1000.0
        ranked = collapse_to_papers(evidence, k)
        scored = rank_metrics(ranked, set(q.relevant), k) if q.relevant else None
        results.append(
            QuestionResult(
                id=q.id,
                topic=q.topic,
                type=q.type,
                question=q.question,
                judged=q.judged,
                ranked_papers=ranked,
                relevant=list(q.relevant),
                metrics=scored,
                top_score=evidence[0].score if evidence else None,
                latency_ms=latency_ms,
            )
        )

    scored_results = [r for r in results if r.metrics is not None]
    by_type: dict[str, TypeSummary] = {}
    for qtype in sorted({r.type for r in scored_results}):
        group = [r.metrics for r in scored_results if r.type == qtype and r.metrics is not None]
        by_type[qtype] = TypeSummary(
            questions=len(group),
            precision=mean(m.precision for m in group),
            recall=mean(m.recall for m in group),
            mrr=mean(m.reciprocal_rank for m in group),
            ndcg=mean(m.ndcg for m in group),
        )
    scored_metrics = [r.metrics for r in scored_results if r.metrics is not None]

    return EvaluationReport(
        benchmark_version=benchmark.version,
        retriever=retriever.name,
        k=k,
        corpus_papers=repository.count_papers(),
        corpus_passages=len(passages),
        generated_at=clock(),
        judged_questions=len(scored_results),
        unjudged_questions=[r.id for r in results if not r.judged],
        mean_precision=mean(m.precision for m in scored_metrics) if scored_metrics else None,
        mean_recall=mean(m.recall for m in scored_metrics) if scored_metrics else None,
        mean_mrr=mean(m.reciprocal_rank for m in scored_metrics) if scored_metrics else None,
        mean_ndcg=mean(m.ndcg for m in scored_metrics) if scored_metrics else None,
        mean_latency_ms=mean(r.latency_ms for r in results),
        by_type=by_type,
        unanswerable=[
            UnanswerableResult(id=r.id, papers_returned=len(r.ranked_papers), top_score=r.top_score)
            for r in results
            if r.type == "unanswerable"
        ],
        results=results,
    )
