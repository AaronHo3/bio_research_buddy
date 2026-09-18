"""Core domain objects.

Rules for this module:

- Nothing here may import from an adapter. Provider response shapes (PubMed XML,
  a future Semantic Scholar JSON, ...) are translated *into* these types by the
  adapter that owns them.
- Absence is explicit. A missing abstract is ``None``, never an empty string,
  so downstream code cannot mistake "we have no text" for "the text is empty".
- Models are frozen. A ``Paper`` is a fact about a source record; nothing
  downstream should mutate it in place.
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

_STRICT = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Source(StrEnum):
    """Literature sources the system knows how to retrieve from."""

    PUBMED = "pubmed"


class PublicationDate(BaseModel):
    """A possibly partial date.

    PubMed records frequently carry only a year, or a year and month. Collapsing
    those to a full ``datetime.date`` would fabricate precision, so the parts are
    stored separately and finer parts require the coarser ones.
    """

    model_config = _STRICT

    year: int = Field(ge=1000, le=9999)
    month: int | None = Field(default=None, ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)

    @model_validator(mode="after")
    def _day_requires_month(self) -> Self:
        if self.day is not None and self.month is None:
            msg = "day cannot be set without month"
            raise ValueError(msg)
        return self


class Paper(BaseModel):
    """A normalized literature record from one source."""

    model_config = _STRICT

    source: Source
    source_id: str = Field(min_length=1, description="Stable ID within the source, e.g. a PMID.")
    title: str = Field(min_length=1)
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: PublicationDate | None = None
    url: str = Field(pattern=r"^https?://")


class Passage(BaseModel):
    """A citable span of one paper, produced by a named chunking strategy.

    ``chunk_id`` is deterministic from the paper identity and position, so the
    same paper always yields the same ids. ``content_hash`` changes when the
    text changes, which lets later stages detect stale embeddings.
    """

    model_config = _STRICT

    source: Source
    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    position: int = Field(ge=0, description="Order within the paper, starting at 0.")
    section: str = Field(min_length=1, description='"title", "abstract", or a section label.')
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of ``text``.")
    strategy: str = Field(min_length=1, description="Name and version of the chunker used.")


class SearchQuery(BaseModel):
    """A user's research question as submitted to the retrieval pipeline."""

    model_config = _STRICT

    raw: str = Field(min_length=1, description="The question exactly as the user typed it.")
    max_results: int = Field(default=10, ge=1, le=100)


class PassageEmbedding(BaseModel):
    """One passage's vector under one named model.

    ``content_hash`` is copied from the passage at embedding time. When a
    passage's text changes its hash changes, so a stored vector whose hash no
    longer matches its passage is stale and must be recomputed. This is what
    makes re-embedding incremental rather than all-or-nothing.
    """

    model_config = _STRICT

    chunk_id: str = Field(min_length=1)
    model: str = Field(min_length=1, description="Embedding model name and version.")
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector: list[float] = Field(min_length=1)

    @property
    def dimension(self) -> int:
        return len(self.vector)


class RetrievedEvidence(BaseModel):
    """One passage returned for a query, with how and how strongly it matched."""

    model_config = _STRICT

    passage: Passage
    rank: int = Field(ge=1)
    score: float = Field(description="The score used for ranking.")
    components: dict[str, float] = Field(
        default_factory=dict, description='Per-method scores, e.g. {"bm25": 3.1}.'
    )
    method: str = Field(min_length=1, description="Retriever name and version.")
