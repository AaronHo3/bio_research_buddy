"""One real embedding model load. Deselected by default; run with ``pytest -m live``.

Downloads the model on first run, then works offline from cache.
"""

import math

import pytest

from bio_research_copilot.adapters.fastembed_model import DEFAULT_MODEL, FastEmbedModel

pytestmark = pytest.mark.live


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm


def test_real_model_embeds_and_separates_related_from_unrelated_text() -> None:
    model = FastEmbedModel(DEFAULT_MODEL)
    assert model.name == DEFAULT_MODEL

    related = "PD-L1 expression predicts which patients respond to checkpoint blockade."
    unrelated = "Optimal water temperature for zebrafish husbandry in laboratory tanks."
    vectors = model.embed_passages([related, unrelated])

    assert len(vectors) == 2
    assert {len(v) for v in vectors} == {384}
    assert all(isinstance(value, float) for value in vectors[0])

    query = model.embed_query("Which biomarkers predict response to immune checkpoint inhibitors?")
    assert len(query) == 384
    assert cosine(query, vectors[0]) > cosine(query, vectors[1])


def test_embedding_is_deterministic() -> None:
    model = FastEmbedModel(DEFAULT_MODEL)
    first = model.embed_passages(["tau phosphorylation drives tangle formation"])
    second = model.embed_passages(["tau phosphorylation drives tangle formation"])
    assert first == second


def test_empty_input_returns_no_vectors() -> None:
    assert FastEmbedModel(DEFAULT_MODEL).embed_passages([]) == []
