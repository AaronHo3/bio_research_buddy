"""Extractive synthesis: build an answer only from sentences that already exist.

Every claim is a sentence copied verbatim from a retrieved passage, so this
cannot fabricate a statement and cannot misattribute one. Faithfulness is a
property of the construction rather than something to be measured and hoped for.

That makes it the right baseline for grounded synthesis. It sets the ceiling on
faithfulness at one hundred per cent and a low floor on answer quality, because
copied sentences do not connect, compare, or resolve disagreement. A language
model has to earn its place by beating the floor without lowering the ceiling.
"""

from bio_research_copilot.domain import Citation, RetrievedEvidence, split_sentences
from bio_research_copilot.retrieval import tokenize

DEFAULT_MAX_CLAIMS = 5


class ExtractiveSynthesizer:
    """Selects the sentence from each passage that best matches the question.

    Args:
        max_claims: Upper bound on claims in one answer.
        min_overlap: Query terms a sentence must share to be claimable. One is
            enough to exclude sentences with nothing to do with the question,
            which is all this is for.
    """

    name = "extractive-v1"

    def __init__(self, *, max_claims: int = DEFAULT_MAX_CLAIMS, min_overlap: int = 1) -> None:
        if max_claims < 1:
            msg = "max_claims must be at least 1"
            raise ValueError(msg)
        if min_overlap < 1:
            msg = "min_overlap must be at least 1"
            raise ValueError(msg)
        self._max_claims = max_claims
        self._min_overlap = min_overlap

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        query_terms = set(tokenize(question))
        if not query_terms:
            return []

        citations: list[Citation] = []
        seen: set[str] = set()
        for item in sorted(evidence, key=lambda e: e.rank):
            best = self._best_sentence(item.passage.text, query_terms)
            if best is None or best in seen:
                continue
            seen.add(best)
            citations.append(Citation(claim=best, evidence_ids=[item.passage.chunk_id]))
            if len(citations) == self._max_claims:
                break
        return citations

    def _best_sentence(self, text: str, query_terms: set[str]) -> str | None:
        """The sentence sharing the most query terms, or None if none qualifies.

        Ties go to the earlier sentence, which in an abstract is usually the more
        general statement and reads better as a standalone claim.
        """
        best: tuple[int, str] | None = None
        for sentence in split_sentences(text):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap < self._min_overlap:
                continue
            if best is None or overlap > best[0]:
                best = (overlap, sentence)
        return best[1] if best else None
