"""Evaluation primitives: metrics, benchmark loading, label agreement. Pure computation."""

from bio_research_copilot.evaluation.agreement import (
    AgreementReport,
    Disagreement,
    compare_labels,
    interpret_kappa,
)
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
    "AgreementReport",
    "Benchmark",
    "BenchmarkQuestion",
    "CorpusManifest",
    "Disagreement",
    "RankMetrics",
    "TopicSnapshot",
    "compare_labels",
    "interpret_kappa",
    "load_benchmark",
    "load_manifest",
    "ndcg_at_k",
    "precision_at_k",
    "rank_metrics",
    "recall_at_k",
    "reciprocal_rank",
]
