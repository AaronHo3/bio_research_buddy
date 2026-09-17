"""The literature search use case.

Deliberately thin today. Its value is the seam: the CLI (and later the API)
call this function, and this function only knows the ``LiteratureSource``
port. Ranking, deduplication, and persistence will attach here without the
entry points changing.
"""

from bio_research_copilot.adapters.literature import LiteratureSource
from bio_research_copilot.domain import Paper, SearchQuery


def search_papers(source: LiteratureSource, question: str, max_results: int) -> list[Paper]:
    """Validate the question into a ``SearchQuery`` and ask the source for papers.

    Raises:
        pydantic.ValidationError: the question is blank or the limit is out of range.
        LiteratureSourceError: the source failed; see the adapter for subclasses.
    """
    query = SearchQuery(raw=question, max_results=max_results)
    return source.search(query)
