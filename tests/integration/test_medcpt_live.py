"""Real MedCPT models. Deselected by default; needs the optional extra.

Run with ``uv sync --extra medcpt`` then ``pytest -m live``.
"""

import pytest

pytestmark = pytest.mark.live

torch = pytest.importorskip("torch", reason="MedCPT needs the optional medcpt extra")

from bio_research_copilot.adapters.medcpt import (  # noqa: E402
    MEDCPT_EMBEDDING_NAME,
    MedCptEmbeddingModel,
    MedCptReranker,
)

QUESTION = "Which biomarkers predict response to immune checkpoint inhibitors?"
RELEVANT = "Navigating established and emerging biomarkers for checkpoint inhibitor therapy."
UNRELATED = "Optimal water temperature for zebrafish husbandry in laboratory tanks."


def test_encoders_load_and_separate_relevant_from_unrelated() -> None:
    model = MedCptEmbeddingModel()
    assert model.name == MEDCPT_EMBEDDING_NAME
    assert model.similarity == "dot"

    vectors = model.embed_passages([UNRELATED, RELEVANT])
    assert {len(v) for v in vectors} == {768}

    query = model.embed_query(QUESTION)
    assert len(query) == 768

    def inner(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert inner(query, vectors[1]) > inner(query, vectors[0])


def test_query_and_passage_encoders_differ() -> None:
    """MedCPT is asymmetric: the same text encoded as a query and as an article differs."""
    model = MedCptEmbeddingModel()
    as_query = model.embed_query(RELEVANT)
    as_passage = model.embed_passages([RELEVANT])[0]
    assert as_query != as_passage


def test_embedding_is_deterministic() -> None:
    model = MedCptEmbeddingModel()
    assert model.embed_passages([RELEVANT]) == model.embed_passages([RELEVANT])


def test_empty_input_returns_nothing() -> None:
    assert MedCptEmbeddingModel().embed_passages([]) == []


def test_cross_encoder_prefers_the_relevant_passage() -> None:
    reranker = MedCptReranker()
    scores = reranker.score(QUESTION, [UNRELATED, RELEVANT])
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_cross_encoder_empty_input() -> None:
    assert MedCptReranker().score(QUESTION, []) == []
