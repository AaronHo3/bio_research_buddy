"""Local embedding models run through ONNX, with no PyTorch dependency.

The model is downloaded once to a cache directory and then runs offline. This
keeps the install small enough for CI and a container, and it keeps evaluation
reproducible: no API key, no network at query time, no provider-side model
changes underneath a recorded result.

Import of the backend is deferred to construction so that importing this module
costs nothing and so a missing or unavailable model surfaces as
``EmbeddingError`` rather than an import failure at start-up.
"""

from pathlib import Path
from typing import Any

from bio_research_copilot.adapters.embedding import EmbeddingError, Similarity

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedModel:
    """An :class:`EmbeddingModel` backed by a local ONNX sentence encoder.

    Args:
        model_name: Hugging Face id of a model the backend supports.
        cache_dir: Where to keep the downloaded model. Defaults to the
            backend's own cache.
    """

    similarity: Similarity = "cosine"
    """These models are trained with normalised vectors and compared by cosine."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised by the install, not tests
            msg = "fastembed is not installed; run 'uv sync' to install project dependencies"
            raise EmbeddingError(msg) from exc
        try:
            self._model: Any = TextEmbedding(
                model_name, cache_dir=str(cache_dir) if cache_dir else None
            )
        except Exception as exc:
            msg = f"could not load embedding model {model_name}: {exc}"
            raise EmbeddingError(msg) from exc
        self.name = model_name

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return [vector.tolist() for vector in self._model.embed(texts)]
        except Exception as exc:
            msg = f"{self.name} failed to embed {len(texts)} passages: {exc}"
            raise EmbeddingError(msg) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vectors = list(self._model.query_embed([text]))
        except Exception as exc:
            msg = f"{self.name} failed to embed the query: {exc}"
            raise EmbeddingError(msg) from exc
        if not vectors:
            msg = f"{self.name} returned no vector for the query"
            raise EmbeddingError(msg)
        return list(vectors[0].tolist())
