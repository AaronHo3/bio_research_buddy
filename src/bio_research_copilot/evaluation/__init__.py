"""Retrieval evaluation: metrics and benchmark loading. Pure computation."""

from bio_research_copilot.evaluation.benchmark import (
    Benchmark,
    BenchmarkQuestion,
    CorpusManifest,
    TopicSnapshot,
    load_benchmark,
    load_manifest,
)
from bio_research_copilot.evaluation.metrics import (
    RankMetrics,
    ndcg_at_k,
    precision_at_k,
    rank_metrics,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "Benchmark",
    "BenchmarkQuestion",
    "CorpusManifest",
    "RankMetrics",
    "TopicSnapshot",
    "load_benchmark",
    "load_manifest",
    "ndcg_at_k",
    "precision_at_k",
    "rank_metrics",
    "recall_at_k",
    "reciprocal_rank",
]
