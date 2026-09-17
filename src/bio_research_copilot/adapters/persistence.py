"""The port for storing papers and the provenance of how they were retrieved.

The application layer depends on ``PaperRepository`` only. SQLite implements it
today; PostgreSQL can later without the services changing.
"""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.domain import Paper, Source


class PersistenceError(Exception):
    """The store could not be opened, read, or written."""


class UpsertResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    inserted: int = Field(ge=0)
    updated: int = Field(ge=0)


class RetrievedPaperRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Source
    source_id: str
    rank: int = Field(ge=1)


class RetrievalRecord(BaseModel):
    """One recorded run of a question against a literature source."""

    model_config = ConfigDict(frozen=True)

    id: int
    question: str
    max_results: int
    retrieved_at: datetime
    papers: list[RetrievedPaperRef]


class PaperRepository(Protocol):
    def upsert_papers(self, papers: list[Paper], *, seen_at: datetime) -> UpsertResult:
        """Insert new papers, refresh existing ones. Idempotent per (source, source_id)."""
        ...

    def get_paper(self, source: Source, source_id: str) -> Paper | None: ...

    def count_papers(self) -> int: ...

    def record_retrieval(
        self, question: str, max_results: int, papers: list[Paper], *, retrieved_at: datetime
    ) -> int:
        """Store which papers a question returned, in rank order. Returns the retrieval id.

        Every paper must already exist in the store.
        """
        ...

    def get_retrieval(self, retrieval_id: int) -> RetrievalRecord | None: ...
