"""The ingestion use case: search, store, and record provenance.

Runs a question against a literature source, upserts the returned papers, and
writes a retrieval record so the run can be inspected later. Re-running the
same question is safe: existing papers are refreshed, not duplicated.
"""

from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from bio_research_copilot.adapters.literature import LiteratureSource
from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.domain import Chunker, SearchQuery, SectionChunker


def utc_now() -> datetime:
    return datetime.now(UTC)


class IngestReport(BaseModel):
    """What one ingestion run did. Printed by the CLI as the manifest."""

    model_config = ConfigDict(frozen=True)

    retrieval_id: int
    question: str
    max_results: int
    retrieved_at: datetime
    papers_fetched: int
    papers_inserted: int
    papers_updated: int
    passages_written: int
    chunking_strategy: str
    paper_ids: list[str]


def ingest_papers(
    source: LiteratureSource,
    repository: PaperRepository,
    question: str,
    max_results: int,
    *,
    chunker: Chunker | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> IngestReport:
    """Search, upsert, chunk, and record. Raises the same errors as the source and repository."""
    chunker = chunker or SectionChunker()
    query = SearchQuery(raw=question, max_results=max_results)
    papers = source.search(query)
    now = clock()
    result = repository.upsert_papers(papers, seen_at=now)
    passages = [passage for paper in papers for passage in chunker.split(paper)]
    written = repository.replace_passages(passages)
    retrieval_id = repository.record_retrieval(
        query.raw, query.max_results, papers, retrieved_at=now
    )
    return IngestReport(
        retrieval_id=retrieval_id,
        question=query.raw,
        max_results=query.max_results,
        retrieved_at=now,
        papers_fetched=len(papers),
        papers_inserted=result.inserted,
        papers_updated=result.updated,
        passages_written=written,
        chunking_strategy=chunker.name,
        paper_ids=[p.source_id for p in papers],
    )
