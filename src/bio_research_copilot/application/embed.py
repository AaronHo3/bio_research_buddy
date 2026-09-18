"""Compute and store vectors for stored passages.

Incremental by design: only passages with no vector for this model, or whose
text changed since it was embedded, are sent to the model. Re-running after
ingesting a few new papers costs a few embeddings rather than a full rebuild.
"""

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.embedding import EmbeddingError, EmbeddingModel
from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.domain import PassageEmbedding

DEFAULT_BATCH_SIZE = 64


class EmbedReport(BaseModel):
    """What one embedding run did."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    dimension: int | None = Field(description="None when nothing needed embedding.")
    passages_total: int
    passages_embedded: int
    passages_already_current: int
    embeddings_stored: int


def embed_corpus(
    repository: PaperRepository,
    model: EmbeddingModel,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbedReport:
    """Embed every passage that needs it. Raises ``EmbeddingError`` if the model fails."""
    if batch_size < 1:
        msg = "batch_size must be at least 1"
        raise ValueError(msg)

    total = repository.count_passages()
    pending = repository.passages_needing_embedding(model.name)

    dimension: int | None = None
    stored = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = model.embed_passages([p.text for p in batch])
        if len(vectors) != len(batch):
            msg = f"{model.name} returned {len(vectors)} vectors for {len(batch)} passages"
            raise EmbeddingError(msg)
        dimensions = {len(v) for v in vectors}
        if len(dimensions) > 1:
            msg = f"{model.name} returned vectors of differing sizes: {sorted(dimensions)}"
            raise EmbeddingError(msg)
        dimension = dimensions.pop()
        stored += repository.upsert_embeddings(
            [
                PassageEmbedding(
                    chunk_id=passage.chunk_id,
                    model=model.name,
                    content_hash=passage.content_hash,
                    vector=vector,
                )
                for passage, vector in zip(batch, vectors, strict=True)
            ]
        )

    return EmbedReport(
        model_name=model.name,
        dimension=dimension,
        passages_total=total,
        passages_embedded=len(pending),
        passages_already_current=total - len(pending),
        embeddings_stored=stored,
    )
