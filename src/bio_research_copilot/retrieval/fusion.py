"""Reciprocal rank fusion: combine several rankings into one.

A passage's fused score is the sum, over the retrievers that returned it, of::

    weight / (constant + rank)

Ranks, not scores. That is the whole point. BM25 scores are unbounded and
depend on the corpus, while cosine similarity lives in [-1, 1], so adding them
directly would let one method's units dominate for no principled reason.
Normalising scores per query instead would make a passage's score depend on
which other passages came back, so adding one document to the corpus could
reorder unrelated results. Positions have neither problem.

The constant flattens the curve near the top. With the conventional value of
60, first place scores 1/61 and second 1/62, so a passage that two retrievers
both rank highly beats one that a single retriever ranks first. Agreement is
rewarded over any one method's confidence, which is exactly the behaviour
wanted when neither method is known to be better.

The cost is that magnitude is discarded. A passage that matches overwhelmingly
and one that matches marginally look identical if they share a rank.
"""

from collections.abc import Sequence

from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery
from bio_research_copilot.retrieval.base import Retriever

DEFAULT_CONSTANT = 60.0
DEFAULT_POOL = 50


class ReciprocalRankFusion:
    """A :class:`Retriever` that fuses the rankings of other retrievers.

    Args:
        retrievers: The rankings to combine. Order affects nothing but the name.
        constant: The ``k`` of the RRF formula. Larger flattens the top further.
        weights: Per-retriever multipliers, defaulting to equal. Tuning these
            requires its own evaluation; they are not tuned here.
        pool: How deep to read each retriever before fusing. Fusing only the
            final k would throw away the agreement evidence that makes this work.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        constant: float = DEFAULT_CONSTANT,
        weights: Sequence[float] | None = None,
        pool: int = DEFAULT_POOL,
    ) -> None:
        if not retrievers:
            msg = "fusion needs at least one retriever"
            raise ValueError(msg)
        if constant <= 0:
            msg = "constant must be positive"
            raise ValueError(msg)
        if pool < 1:
            msg = "pool must be at least 1"
            raise ValueError(msg)
        if weights is not None and len(weights) != len(retrievers):
            msg = f"got {len(weights)} weights for {len(retrievers)} retrievers"
            raise ValueError(msg)

        self._retrievers = list(retrievers)
        self._constant = constant
        self._weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        self._pool = pool
        self.name = f"rrf({'+'.join(r.name for r in self._retrievers)})"

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        depth = min(100, max(self._pool, query.max_results))
        deep_query = SearchQuery(raw=query.raw, max_results=depth)

        contributions: dict[str, dict[str, float]] = {}
        passages: dict[str, Passage] = {}
        for retriever, weight in zip(self._retrievers, self._weights, strict=True):
            for item in retriever.search(deep_query):
                chunk_id = item.passage.chunk_id
                passages.setdefault(chunk_id, item.passage)
                contributions.setdefault(chunk_id, {})[retriever.name] = weight / (
                    self._constant + item.rank
                )

        ranked = sorted(
            contributions.items(), key=lambda entry: (-sum(entry[1].values()), entry[0])
        )
        return [
            RetrievedEvidence(
                passage=passages[chunk_id],
                rank=rank,
                score=sum(components.values()),
                components=components,
                method=self.name,
            )
            for rank, (chunk_id, components) in enumerate(ranked[: query.max_results], start=1)
        ]
