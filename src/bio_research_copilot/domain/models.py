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


class Citation(BaseModel):
    """One claim in an answer, with the evidence it rests on.

    ``evidence_ids`` are passage chunk ids. A citation with no evidence is not
    representable: a claim the system cannot support must not be made at all.
    """

    model_config = _STRICT

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class ResearchAnswer(BaseModel):
    """A question, what the system concluded, and what it rested that on.

    ``abstained`` is not the same as an empty answer. Abstaining is a positive
    statement that the retrieved evidence did not support a response, which the
    product specification requires the interface to show.
    """

    model_config = _STRICT

    question: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list["RetrievedEvidence"] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    abstained: bool = False
    retrieval_method: str = Field(min_length=1)
    synthesis_method: str = Field(min_length=1)

    @model_validator(mode="after")
    def _claims_require_evidence(self) -> Self:
        """Every cited id must be in this answer's own evidence.

        This is the Stage 5 gate from the project specification: no displayed
        citation may refer to evidence that is not present. Enforcing it in the
        model rather than in a check means an answer violating it cannot be
        constructed at all, whatever produced it.
        """
        if self.abstained and self.citations:
            msg = "an abstained answer cannot carry citations"
            raise ValueError(msg)
        available = {item.passage.chunk_id for item in self.evidence}
        for citation in self.citations:
            unknown = [ref for ref in citation.evidence_ids if ref not in available]
            if unknown:
                msg = f"citation refers to evidence not in this answer: {sorted(unknown)}"
                raise ValueError(msg)
        return self


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


ResearchAnswer.model_rebuild()
