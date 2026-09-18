"""The port every embedding provider must implement.

Queries and passages are embedded by separate methods on purpose. Retrieval
models are often asymmetric: some prepend an instruction to queries only, and
some, such as NCBI's MedCPT, ship two different encoders. A single ``embed``
method would force those providers to guess which side they were being asked
for, and guessing wrong quietly degrades retrieval rather than failing.
"""

from typing import Protocol


class EmbeddingError(Exception):
    """The model could not be loaded or could not embed the given text."""


class EmbeddingModel(Protocol):
    name: str
    """Model name and version, stored with every vector so provenance is explicit."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Vectors for stored text, in the order given.

        Raises:
            EmbeddingError: the model is unavailable or the text could not be encoded.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Vector for a search query, which some models encode differently."""
        ...
