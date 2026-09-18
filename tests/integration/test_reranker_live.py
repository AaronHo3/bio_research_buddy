"""One real cross-encoder load. Deselected by default; run with ``pytest -m live``."""

import pytest

from bio_research_copilot.adapters.fastembed_reranker import (
    DEFAULT_RERANK_MODEL,
    FastEmbedReranker,
)

pytestmark = pytest.mark.live

QUESTION = "Which biomarkers predict response to immune checkpoint inhibitors?"


def test_real_reranker_prefers_the_relevant_passage() -> None:
    reranker = FastEmbedReranker(DEFAULT_RERANK_MODEL)
    assert reranker.name == DEFAULT_RERANK_MODEL

    relevant = (
        "PD-L1 expression and tumour mutational burden predict which patients"
        " respond to checkpoint blockade."
    )
    unrelated = "Optimal water temperature for zebrafish husbandry in laboratory tanks."
    scores = reranker.score(QUESTION, [unrelated, relevant])

    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_scoring_is_deterministic() -> None:
    reranker = FastEmbedReranker(DEFAULT_RERANK_MODEL)
    text = ["Tau phosphorylation drives tangle formation in neurons."]
    assert reranker.score(QUESTION, text) == reranker.score(QUESTION, text)


def test_empty_input_returns_no_scores() -> None:
    assert FastEmbedReranker(DEFAULT_RERANK_MODEL).score(QUESTION, []) == []
