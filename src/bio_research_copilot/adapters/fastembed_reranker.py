"""Local cross-encoder reranking through ONNX, with no PyTorch dependency.

Same runtime and the same tradeoffs as the embedding adapter: downloaded once,
then offline, so a recorded evaluation result cannot shift underneath you.
"""

from pathlib import Path
from typing import Any

from bio_research_copilot.adapters.reranking import RerankerError

DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class FastEmbedReranker:
    """A :class:`Reranker` backed by a local ONNX cross-encoder.

    Args:
        model_name: Hugging Face id of a cross-encoder the backend supports.
        cache_dir: Where to keep the downloaded model.
    """

    def __init__(
        self, model_name: str = DEFAULT_RERANK_MODEL, cache_dir: Path | None = None
    ) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - an install problem, not a test path
            msg = "fastembed is not installed; run 'uv sync' to install project dependencies"
            raise RerankerError(msg) from exc
        try:
            self._model: Any = TextCrossEncoder(
                model_name, cache_dir=str(cache_dir) if cache_dir else None
            )
        except Exception as exc:
            msg = f"could not load reranking model {model_name}: {exc}"
            raise RerankerError(msg) from exc
        self.name = model_name

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        try:
            scores = [float(value) for value in self._model.rerank(query, texts)]
        except Exception as exc:
            msg = f"{self.name} failed to score {len(texts)} passages: {exc}"
            raise RerankerError(msg) from exc
        if len(scores) != len(texts):
            msg = f"{self.name} returned {len(scores)} scores for {len(texts)} passages"
            raise RerankerError(msg)
        return scores
