"""The end-to-end use case: a question in, a cited answer or an abstention out.

This is the first point where the product the specification describes actually
exists. It retrieves, budgets the evidence, synthesises claims from it, and
returns an answer in which every claim points at a passage the caller can open.

Abstention is a first-class outcome, not an error. The specification requires
the system to say when evidence is missing rather than answer anyway, and there
are three ways that happens here: retrieval found nothing, retrieval found
things that scored below a configured floor, or synthesis could not make a claim
from what it found.
"""

from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.application.query import RetrieverFactory, build_bm25
from bio_research_copilot.domain import ResearchAnswer, RetrievedEvidence, SearchQuery
from bio_research_copilot.synthesis import (
    DEFAULT_CHAR_BUDGET,
    ExtractiveSynthesizer,
    SynthesisProvider,
    select_evidence,
)

DEFAULT_EVIDENCE_LIMIT = 8

ABSTAIN_NO_RESULTS = "Retrieval returned no passages for this question."
ABSTAIN_BELOW_FLOOR = "The best retrieved passage scored below the configured relevance floor."
ABSTAIN_NO_CLAIM = "Retrieved passages did not contain a sentence that addresses the question."

SCOPE_LIMITATION = "Evidence is limited to titles and abstracts of stored papers, not full text."


def answer_question(
    repository: PaperRepository,
    question: str,
    *,
    build_retriever: RetrieverFactory = build_bm25,
    synthesizer: SynthesisProvider | None = None,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    min_score: float | None = None,
) -> ResearchAnswer:
    """Retrieve, ground, and answer. Never returns a claim it cannot cite.

    Args:
        min_score: Abstain when the top retrieved score is below this. Scores
            are on unrelated scales between retrieval methods, so a floor set
            for one is meaningless for another and none is applied by default.

    Raises:
        pydantic.ValidationError: the question is blank.
        PersistenceError, EmbeddingError, RerankerError: a dependency failed.
        SynthesisError: the provider failed outright.
    """
    provider = synthesizer or ExtractiveSynthesizer()
    query = SearchQuery(raw=question, max_results=evidence_limit)
    retriever = build_retriever(repository)
    evidence = retriever.search(query)

    def abstain(reason: str) -> ResearchAnswer:
        return ResearchAnswer(
            question=query.raw,
            citations=[],
            evidence=evidence,
            limitations=[reason, SCOPE_LIMITATION],
            abstained=True,
            retrieval_method=retriever.name,
            synthesis_method=provider.name,
        )

    if not evidence:
        return abstain(ABSTAIN_NO_RESULTS)
    if min_score is not None and evidence[0].score < min_score:
        return abstain(ABSTAIN_BELOW_FLOOR)

    selected = select_evidence(evidence, char_budget=char_budget)
    citations = provider.synthesize(query.raw, selected)
    if not citations:
        return abstain(ABSTAIN_NO_CLAIM)

    return ResearchAnswer(
        question=query.raw,
        citations=citations,
        # Only what synthesis could see, so a citation can never point outside it.
        evidence=selected,
        limitations=_limitations(selected, dropped=len(evidence) - len(selected)),
        abstained=False,
        retrieval_method=retriever.name,
        synthesis_method=provider.name,
    )


def _limitations(selected: list[RetrievedEvidence], dropped: int) -> list[str]:
    papers = {item.passage.source_id for item in selected}
    notes = [
        f"Based on {len(selected)} passages from {len(papers)} papers.",
        SCOPE_LIMITATION,
    ]
    if dropped:
        notes.append(f"{dropped} lower-ranked passages did not fit the evidence budget.")
    if len(papers) == 1:
        notes.append("All evidence comes from a single paper, so it is not corroborated.")
    return notes


def paper_ids_cited(answer: ResearchAnswer) -> list[str]:
    """Source ids behind an answer's citations, in citation order."""
    by_chunk = {item.passage.chunk_id: item.passage for item in answer.evidence}
    ordered: list[str] = []
    for citation in answer.citations:
        for ref in citation.evidence_ids:
            passage = by_chunk.get(ref)
            if passage and passage.source_id not in ordered:
                ordered.append(passage.source_id)
    return ordered
