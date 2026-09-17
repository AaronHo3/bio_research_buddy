"""The local retrieval use case: rank stored passages for a question.

Builds the retriever from the whole stored corpus on every call. That costs
milliseconds for thousands of passages and keeps the index always in sync
with the store; a persistent index is a later optimisation, if measured.
"""

from collections.abc import Callable

from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery
from bio_research_copilot.retrieval import BM25Index, Retriever

RetrieverFactory = Callable[[list[Passage]], Retriever]


def build_bm25(passages: list[Passage]) -> Retriever:
    return BM25Index(passages)


def query_passages(
    repository: PaperRepository,
    question: str,
    max_results: int,
    *,
    build_retriever: RetrieverFactory = build_bm25,
) -> list[RetrievedEvidence]:
    """Validate the question, index the stored passages, and return ranked evidence.

    Raises:
        pydantic.ValidationError: the question is blank or the limit is out of range.
        PersistenceError: the store could not be read.
    """
    query = SearchQuery(raw=question, max_results=max_results)
    retriever = build_retriever(repository.all_passages())
    return retriever.search(query)
