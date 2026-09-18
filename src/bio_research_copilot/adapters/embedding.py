"""The port every embedding provider must implement.

Queries and passages are embedded by separate methods on purpose. Retrieval
models are often asymmetric: some prepend an instruction to queries only, and
some, such as NCBI's MedCPT, ship two different encoders. A single ``embed``
method would force those providers to guess which side they were being asked
for, and guessing wrong quietly degrades retrieval rather than failing.
"""

from typing import Literal, Protocol

Similarity = Literal["cosine", "dot"]
"""How two of a model's vectors should be compared.

This belongs to the model, not to the index. Most sentence encoders are trained
with normalised vectors and are compared by cosine. Some, such as MedCPT, are
trained on the raw inner product, where vector magnitude carries signal;
normalising those before comparing throws that signal away and quietly degrades
retrieval. A model that knows how it was trained should say so.
"""


class EmbeddingError(Exception):
    """The model could not be loaded or could not embed the given text."""


class EmbeddingModel(Protocol):
    name: str
    """Model name and version, stored with every vector so provenance is explicit."""

    similarity: Similarity
    """How the index should compare this model's vectors."""

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Vectors for stored text, in the order given.

        Raises:
            EmbeddingError: the model is unavailable or the text could not be encoded.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Vector for a search query, which some models encode differently."""
        ...
