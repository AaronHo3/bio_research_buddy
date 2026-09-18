"""Test doubles shared across unit tests."""

import hashlib
import math

from bio_research_copilot.adapters.embedding import EmbeddingError


class FakeClock:
    """A clock that only advances when told to, or when something 'sleeps'."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEmbeddingModel:
    """A deterministic stand-in for a real encoder, with no model download.

    Vectors come from a hash of the text, so identical text always yields an
    identical vector and different text yields a different one. That is enough
    to test storage, staleness, and batching. It says nothing about meaning, so
    retrieval quality is never asserted against this.
    """

    def __init__(self, name: str = "fake-v1", dimension: int = 8) -> None:
        self.name = name
        self.dimension = dimension
        self.batches: list[list[str]] = []
        self.queries: list[str] = []
        self.fail_with: Exception | None = None
        self.wrong_count: bool = False

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        self.batches.append(list(texts))
        vectors = [self._vector(t) for t in texts]
        return vectors[:-1] if self.wrong_count and vectors else vectors

    def embed_query(self, text: str) -> list[float]:
        if self.fail_with:
            raise self.fail_with
        self.queries.append(text)
        return self._vector(text)


def failing_model(message: str = "model unavailable") -> FakeEmbeddingModel:
    model = FakeEmbeddingModel()
    model.fail_with = EmbeddingError(message)
    return model
