"""The port every retriever implements."""

from typing import Protocol

from bio_research_copilot.domain import RetrievedEvidence, SearchQuery


class Retriever(Protocol):
    name: str

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        """Return up to ``query.max_results`` passages, best first, ranks starting at 1."""
        ...
