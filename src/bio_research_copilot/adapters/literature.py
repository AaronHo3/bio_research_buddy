"""The port that every literature provider must implement.

The application layer depends on this protocol, never on a concrete provider,
so a provider can be swapped or faked in tests without touching business logic.
"""

from typing import Protocol

from bio_research_copilot.domain import Paper, SearchQuery


class LiteratureSourceError(Exception):
    """Base class for provider failures. Callers can catch this alone."""


class LiteratureSourceUnavailableError(LiteratureSourceError):
    """The provider could not be reached or refused to serve (timeout, 5xx, rate limit)."""


class LiteratureSourceParseError(LiteratureSourceError):
    """The provider answered, but the response could not be translated into domain objects."""


class LiteratureSource(Protocol):
    """Anything that can turn a query into normalized papers."""

    def search(self, query: SearchQuery, term: str | None = None) -> list[Paper]:
        """Return up to ``query.max_results`` papers, best match first.

        Args:
            query: The user's question and how many papers to return.
            term: A provider-specific query string prepared by a caller that
                knows how to write one. A source that ignores it still satisfies
                this contract, which keeps query planning optional rather than
                required.

        Raises:
            LiteratureSourceUnavailableError: the provider did not answer usefully.
            LiteratureSourceParseError: the answer could not be understood.
        """
        ...
