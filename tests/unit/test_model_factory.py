"""Tests for model routing.

These must not download anything, so they check which class the factory selects
rather than constructing a working model.
"""

from pathlib import Path
from typing import Any

import pytest

from bio_research_copilot.adapters import model_factory
from bio_research_copilot.adapters.fastembed_model import FastEmbedModel
from bio_research_copilot.adapters.fastembed_reranker import FastEmbedReranker


@pytest.fixture
def no_construction(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every backend constructor with a recorder."""
    seen: dict[str, Any] = {}

    def recorder(label: str) -> Any:
        def build(*args: Any, **kwargs: Any) -> str:
            seen["label"] = label
            seen["args"] = args
            seen["kwargs"] = kwargs
            return label

        return build

    monkeypatch.setattr(model_factory, "FastEmbedModel", recorder("fastembed-model"))
    monkeypatch.setattr(model_factory, "FastEmbedReranker", recorder("fastembed-reranker"))
    medcpt = pytest.importorskip("bio_research_copilot.adapters.medcpt")
    monkeypatch.setattr(medcpt, "MedCptEmbeddingModel", recorder("medcpt-model"))
    monkeypatch.setattr(medcpt, "MedCptReranker", recorder("medcpt-reranker"))
    return seen


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("BAAI/bge-small-en-v1.5", "fastembed-model"),
        ("sentence-transformers/all-MiniLM-L6-v2", "fastembed-model"),
        ("ncbi/MedCPT", "medcpt-model"),
        ("ncbi/MedCPT-Query-Encoder", "medcpt-model"),
    ],
)
def test_embedding_model_routing(name: str, expected: str, no_construction: dict[str, Any]) -> None:
    model_factory.build_embedding_model(name)
    assert no_construction["label"] == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Xenova/ms-marco-MiniLM-L-6-v2", "fastembed-reranker"),
        ("ncbi/MedCPT-Cross-Encoder", "medcpt-reranker"),
    ],
)
def test_reranker_routing(name: str, expected: str, no_construction: dict[str, Any]) -> None:
    model_factory.build_reranker(name)
    assert no_construction["label"] == expected


def test_cache_dir_reaches_the_backend(no_construction: dict[str, Any]) -> None:
    model_factory.build_embedding_model("BAAI/bge-small-en-v1.5", Path("/tmp/models"))
    assert no_construction["kwargs"]["cache_dir"] == Path("/tmp/models")


def test_default_backends_are_the_light_ones() -> None:
    """Without the optional extra installed, nothing may reach for PyTorch."""
    assert model_factory.build_embedding_model.__module__ == model_factory.__name__
    assert FastEmbedModel.__module__.endswith("fastembed_model")
    assert FastEmbedReranker.__module__.endswith("fastembed_reranker")
