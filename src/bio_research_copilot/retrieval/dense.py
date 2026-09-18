"""Dense retrieval: rank passages by cosine similarity to the query's vector.

The whole method is one matrix multiply. Every stored vector is normalised to
unit length once at construction, the query vector is normalised at search
time, and the dot product of two unit vectors is their cosine similarity.

Search is brute force over every vector. At this corpus size that is a fraction
of a millisecond and it is exact. An approximate index trades recall for speed
and is only worth its complexity at a scale this project has not reached; when
it is, it becomes another ``Retriever`` and the evaluation compares the two.

Unlike BM25, dense retrieval returns a score for *every* passage, because two
unrelated texts still have some cosine similarity. It therefore never returns
an empty result, which matters for questions the corpus cannot answer.
"""

from collections.abc import Sequence

import numpy as np

from bio_research_copilot.adapters.embedding import EmbeddingError, EmbeddingModel
from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery


class DenseIndex:
    """An :class:`EmbeddingModel`-backed :class:`Retriever` over stored vectors.

    Args:
        entries: Each passage paired with the vector computed from its text.
        model: Used only to embed the query; passage vectors come from storage.
    """

    def __init__(
        self,
        entries: Sequence[tuple[Passage, Sequence[float]]],
        model: EmbeddingModel,
    ) -> None:
        self._model = model
        self._similarity = getattr(model, "similarity", "cosine")
        self.name = f"dense:{model.name}"
        # Stable order so equal scores always tie-break the same way.
        ordered = sorted(entries, key=lambda pair: pair[0].chunk_id)
        self._passages = [passage for passage, _ in ordered]
        if ordered:
            dimensions = {len(vector) for _, vector in ordered}
            if len(dimensions) > 1:
                msg = f"stored vectors have differing sizes: {sorted(dimensions)}"
                raise EmbeddingError(msg)
            matrix = np.array([list(v) for _, v in ordered], dtype=np.float32)
            # Normalising only makes sense for a model trained that way; see Similarity.
            self._matrix = _unit_rows(matrix) if self._similarity == "cosine" else matrix
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        if not self._passages:
            return []
        vector = np.array(self._model.embed_query(query.raw), dtype=np.float32)
        if vector.shape[0] != self._matrix.shape[1]:
            msg = (
                f"{self._model.name} produced a {vector.shape[0]}-dimensional query vector"
                f" but stored vectors are {self._matrix.shape[1]}-dimensional;"
                " re-run embed for this model"
            )
            raise EmbeddingError(msg)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            msg = f"{self._model.name} produced an all-zero query vector"
            raise EmbeddingError(msg)
        scores = self._matrix @ (vector / norm if self._similarity == "cosine" else vector)

        # Descending score, then chunk_id, which the stable sort already gives us.
        order = np.argsort(-scores, kind="stable")[: query.max_results]
        return [
            RetrievedEvidence(
                passage=self._passages[int(i)],
                rank=rank,
                score=float(scores[int(i)]),
                components={self._similarity: float(scores[int(i)])},
                method=self.name,
            )
            for rank, i in enumerate(order, start=1)
        ]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)
