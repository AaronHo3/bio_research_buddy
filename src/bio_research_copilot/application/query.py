"""The local retrieval use case: rank stored passages for a question.

A retriever is built from the repository on every call. That costs milliseconds
at this corpus size and guarantees the index never drifts from the store; a
persistent index is a later optimisation, if measured.
"""

from collections.abc import Callable

from bio_research_copilot.adapters.embedding import EmbeddingError, EmbeddingModel
from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.adapters.reranking import Reranker
from bio_research_copilot.domain import RetrievedEvidence, SearchQuery
from bio_research_copilot.retrieval import (
    BM25Index,
    DenseIndex,
    ReciprocalRankFusion,
    RerankingRetriever,
    Retriever,
)
from bio_research_copilot.retrieval.rerank import DEFAULT_POOL as DEFAULT_RERANK_POOL

RetrieverFactory = Callable[[PaperRepository], Retriever]


def build_bm25(repository: PaperRepository) -> Retriever:
    return BM25Index(repository.all_passages())


def build_dense(repository: PaperRepository, model: EmbeddingModel) -> Retriever:
    """Pair stored passages with their vectors for this model.

    Raises:
        EmbeddingError: no passage has a vector for this model, or some do not.
            Both mean ``embed`` has not been run (or not finished) for it, and
            silently searching a partial corpus would understate recall.
    """
    passages = repository.all_passages()
    vectors = {e.chunk_id: e.vector for e in repository.get_embeddings(model.name)}
    missing = [p.chunk_id for p in passages if p.chunk_id not in vectors]
    if passages and missing:
        msg = (
            f"{len(missing)} of {len(passages)} passages have no vector for {model.name};"
            f" run 'bio-copilot embed --model {model.name}' first"
        )
        raise EmbeddingError(msg)
    return DenseIndex([(p, vectors[p.chunk_id]) for p in passages], model)


def build_hybrid(repository: PaperRepository, model: EmbeddingModel) -> Retriever:
    """Fuse the lexical and dense rankings. Requires vectors, like ``build_dense``."""
    return ReciprocalRankFusion([build_bm25(repository), build_dense(repository, model)])


def with_reranking(
    factory: RetrieverFactory, reranker: Reranker, *, pool: int = DEFAULT_RERANK_POOL
) -> RetrieverFactory:
    """Wrap any retriever factory so a cross-encoder reorders its shortlist."""
    return lambda repository: RerankingRetriever(factory(repository), reranker, pool=pool)


def query_passages(
    repository: PaperRepository,
    question: str,
    max_results: int,
    *,
    build_retriever: RetrieverFactory = build_bm25,
) -> list[RetrievedEvidence]:
    """Validate the question, build the retriever, and return ranked evidence.

    Raises:
        pydantic.ValidationError: the question is blank or the limit is out of range.
        PersistenceError: the store could not be read.
        EmbeddingError: a dense retriever was asked for but vectors are missing.
    """
    query = SearchQuery(raw=question, max_results=max_results)
    return build_retriever(repository).search(query)
