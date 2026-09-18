"""The port every reranker must implement.

A reranker is a cross-encoder: it reads the query and one passage together in a
single forward pass and returns how well they answer each other. That is a
different operation from embedding. A bi-encoder turns each side into a vector
independently, which makes the passage side precomputable and the comparison
cheap, at the cost of never letting the two texts interact. A cross-encoder
lets them interact and therefore cannot precompute anything, so its cost is one
model call per candidate.

That cost is why reranking runs over a shortlist rather than a corpus.
"""

from typing import Protocol


class RerankerError(Exception):
    """The reranker could not be loaded or could not score the given texts."""


class Reranker(Protocol):
    name: str
    """Model name and version, recorded on every result for provenance."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to the query, in the order given.

        Scores are comparable within one call and carry no fixed scale, so they
        order a shortlist and mean nothing on their own.

        Raises:
            RerankerError: the model is unavailable or scoring failed.
        """
        ...
