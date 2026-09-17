"""Ranked-retrieval metrics with binary relevance.

All functions take a ranked list of item ids (best first) and the set of ids
judged relevant. They are defined only when at least one item is relevant;
callers handle the unanswerable case separately.

- precision@k: of the top k returned, what fraction is relevant.
- recall@k: of everything relevant, what fraction appears in the top k.
- reciprocal rank: 1 / position of the first relevant item, 0 if none.
- nDCG@k: discounted cumulative gain, normalised by the best possible ordering.
  A relevant item at rank i earns 1 / log2(i + 1), so early hits count more.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _check(relevant: set[str], k: int) -> None:
    if not relevant:
        msg = "metrics are undefined when nothing is relevant"
        raise ValueError(msg)
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    _check(relevant, 1)
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    _check(relevant, k)
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg


@dataclass(frozen=True)
class RankMetrics:
    k: int
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float


def rank_metrics(ranked: Sequence[str], relevant: set[str], k: int) -> RankMetrics:
    return RankMetrics(
        k=k,
        precision=precision_at_k(ranked, relevant, k),
        recall=recall_at_k(ranked, relevant, k),
        reciprocal_rank=reciprocal_rank(ranked, relevant),
        ndcg=ndcg_at_k(ranked, relevant, k),
    )
