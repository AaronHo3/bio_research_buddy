"""Two-stage retrieval: a cheap retriever proposes, a cross-encoder disposes.

The wrapped retriever supplies a shortlist, the reranker scores every candidate
against the query, and the shortlist is reordered. Nothing is added to or
removed from the candidate set, so reranking can only change the order. Recall
at the shortlist depth is therefore an upper bound on what this can achieve,
and that is the division of labour the evaluation of fusion pointed at: fusion
assembles a better candidate set, reranking orders it.

The shortlist depth is the one real dial. Too shallow and a relevant passage
never reaches the reranker; too deep and the cost grows with no benefit, since
cost is one model call per candidate.
"""

from bio_research_copilot.adapters.reranking import Reranker
from bio_research_copilot.domain import RetrievedEvidence, SearchQuery
from bio_research_copilot.retrieval.base import Retriever

DEFAULT_POOL = 25


class RerankingRetriever:
    """A :class:`Retriever` that reorders another retriever's shortlist.

    Args:
        base: Supplies the candidates. Its scores are kept for comparison.
        reranker: Scores each candidate against the query.
        pool: How many candidates to rerank. Capped by the query limit.
    """

    def __init__(self, base: Retriever, reranker: Reranker, *, pool: int = DEFAULT_POOL) -> None:
        if pool < 1:
            msg = "pool must be at least 1"
            raise ValueError(msg)
        self._base = base
        self._reranker = reranker
        self._pool = pool
        self.name = f"rerank:{reranker.name}({base.name})"

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        depth = min(100, max(self._pool, query.max_results))
        candidates = self._base.search(SearchQuery(raw=query.raw, max_results=depth))
        if not candidates:
            return []

        scores = self._reranker.score(query.raw, [c.passage.text for c in candidates])
        # Highest score first, then chunk_id so equal scores are deterministic.
        ordered = sorted(
            zip(scores, candidates, strict=True),
            key=lambda pair: (-pair[0], pair[1].passage.chunk_id),
        )
        return [
            RetrievedEvidence(
                passage=candidate.passage,
                rank=rank,
                score=score,
                # Keep what the first stage thought, so a regression can be traced
                # to the shortlist or to the reordering.
                components={**candidate.components, f"rerank:{self._reranker.name}": score},
                method=self.name,
            )
            for rank, (score, candidate) in enumerate(ordered[: query.max_results], start=1)
        ]
