"""NCBI's MedCPT, a retrieval stack trained on PubMed search logs.

Where the general-purpose encoder learned relatedness from web text, MedCPT
learned it from what biomedical researchers actually clicked on after issuing a
query. It ships three models and this module wraps two of them:

- A bi-encoder pair. Queries and articles go through *different* models, trained
  together. A question and an abstract are different kinds of text, and MedCPT
  encodes each with a network specialised for it. The port's separate
  ``embed_query`` and ``embed_passages`` methods exist for exactly this.
- A cross-encoder for reranking, which reads a query and an article together.

Two details matter for correctness. MedCPT scores by the raw inner product of
the two ``[CLS]`` vectors rather than cosine, so magnitude carries signal and
the vectors must not be normalised; that is declared through ``similarity``.
And the encoders have different input budgets: short for queries, long for
articles.

This module needs PyTorch, so it is an optional dependency group and imports
are deferred until construction.
"""

from pathlib import Path
from typing import Any

from bio_research_copilot.adapters.embedding import EmbeddingError, Similarity
from bio_research_copilot.adapters.reranking import RerankerError

QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"

MEDCPT_EMBEDDING_NAME = "ncbi/MedCPT"
"""One logical name for the encoder pair, since vectors are stored per model."""

QUERY_MAX_TOKENS = 64
ARTICLE_MAX_TOKENS = 512


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        msg = (
            "MedCPT needs PyTorch, which is an optional dependency;"
            " install it with 'uv sync --extra medcpt'"
        )
        raise EmbeddingError(msg) from exc
    return torch


class MedCptEmbeddingModel:
    """MedCPT's bi-encoder pair behind the :class:`EmbeddingModel` port.

    Args:
        cache_dir: Where to keep the downloaded weights.
        device: Torch device string. Defaults to CPU.
    """

    name = MEDCPT_EMBEDDING_NAME
    similarity: Similarity = "dot"
    """MedCPT is trained on the raw inner product; normalising would discard signal."""

    def __init__(self, cache_dir: Path | None = None, device: str = "cpu") -> None:
        self._torch = _load_torch()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            msg = "MedCPT needs transformers; install it with 'uv sync --extra medcpt'"
            raise EmbeddingError(msg) from exc

        folder = str(cache_dir) if cache_dir else None
        try:
            self._query_tokenizer = AutoTokenizer.from_pretrained(QUERY_ENCODER, cache_dir=folder)
            self._query_model = AutoModel.from_pretrained(QUERY_ENCODER, cache_dir=folder).eval()
            self._article_tokenizer = AutoTokenizer.from_pretrained(
                ARTICLE_ENCODER, cache_dir=folder
            )
            self._article_model = AutoModel.from_pretrained(
                ARTICLE_ENCODER, cache_dir=folder
            ).eval()
        except Exception as exc:
            msg = f"could not load MedCPT encoders: {exc}"
            raise EmbeddingError(msg) from exc
        self._device = device
        self._query_model.to(device)
        self._article_model.to(device)

    def _encode(
        self, model: Any, tokenizer: Any, texts: list[str], limit: int
    ) -> list[list[float]]:
        encoded = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt", max_length=limit
        ).to(self._device)
        with self._torch.no_grad():
            # MedCPT represents a text by its [CLS] token, the first position.
            vectors = model(**encoded).last_hidden_state[:, 0, :]
        return [[float(value) for value in row] for row in vectors.cpu().tolist()]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return self._encode(
                self._article_model, self._article_tokenizer, texts, ARTICLE_MAX_TOKENS
            )
        except Exception as exc:
            msg = f"{self.name} failed to embed {len(texts)} passages: {exc}"
            raise EmbeddingError(msg) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            vectors = self._encode(
                self._query_model, self._query_tokenizer, [text], QUERY_MAX_TOKENS
            )
        except Exception as exc:
            msg = f"{self.name} failed to embed the query: {exc}"
            raise EmbeddingError(msg) from exc
        return vectors[0]


class MedCptReranker:
    """MedCPT's cross-encoder behind the :class:`Reranker` port."""

    name = CROSS_ENCODER

    def __init__(self, cache_dir: Path | None = None, device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            msg = (
                "MedCPT needs PyTorch and transformers, an optional dependency group;"
                " install them with 'uv sync --extra medcpt'"
            )
            raise RerankerError(msg) from exc
        self._torch = torch

        folder = str(cache_dir) if cache_dir else None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER, cache_dir=folder)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                CROSS_ENCODER, cache_dir=folder
            ).eval()
        except Exception as exc:
            msg = f"could not load {CROSS_ENCODER}: {exc}"
            raise RerankerError(msg) from exc
        self._device = device
        self._model.to(device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        try:
            encoded = self._tokenizer(
                [[query, text] for text in texts],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=ARTICLE_MAX_TOKENS,
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(dim=-1)
            scores = [float(value) for value in logits.cpu().reshape(-1).tolist()]
        except Exception as exc:
            msg = f"{self.name} failed to score {len(texts)} passages: {exc}"
            raise RerankerError(msg) from exc
        if len(scores) != len(texts):
            msg = f"{self.name} returned {len(scores)} scores for {len(texts)} passages"
            raise RerankerError(msg)
        return scores
