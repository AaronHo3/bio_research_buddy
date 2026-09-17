"""Tests for ranking metrics, each checked against a hand computation."""

import math

import pytest

from bio_research_copilot.evaluation import (
    ndcg_at_k,
    precision_at_k,
    rank_metrics,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_precision_counts_relevant_in_top_k() -> None:
    assert precision_at_k(RANKED, {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(RANKED, {"e"}, 3) == 0.0


def test_precision_divides_by_k_even_when_fewer_returned() -> None:
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(1 / 5)


def test_recall_divides_by_number_relevant() -> None:
    assert recall_at_k(RANKED, {"a", "c", "z"}, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"a", "c", "z"}, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, {"a"}, 1) == 1.0


def test_reciprocal_rank_uses_first_hit() -> None:
    assert reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(RANKED, {"a"}) == 1.0
    assert reciprocal_rank(RANKED, {"z"}) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0


def test_ndcg_perfect_ordering_is_one() -> None:
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["a", "x", "y"], {"a"}, 3)
    late = ndcg_at_k(["x", "y", "a"], {"a"}, 3)
    assert early == 1.0
    assert late == pytest.approx((1 / math.log2(4)) / 1.0)
    assert early > late


def test_ndcg_hand_computation() -> None:
    # hits at ranks 1 and 3 of 3; two relevant items -> ideal has hits at 1 and 2.
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_is_capped_at_k() -> None:
    # 5 relevant but k=2: ideal can hold only 2 hits, so 2 hits in top 2 is perfect.
    assert ndcg_at_k(["a", "b"], {"a", "b", "c", "d", "e"}, 2) == pytest.approx(1.0)


def test_no_hits_gives_zero_everywhere() -> None:
    m = rank_metrics(RANKED, {"z"}, 3)
    assert (m.precision, m.recall, m.reciprocal_rank, m.ndcg) == (0.0, 0.0, 0.0, 0.0)


def test_undefined_when_nothing_relevant() -> None:
    with pytest.raises(ValueError, match="undefined"):
        precision_at_k(RANKED, set(), 3)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(RANKED, {"a"}, 0)
