"""Turn an evaluated synthesis report into data a web page can render.

The demo shows real model output without calling a model. Every answer here was
produced by a measured evaluation run and is replayed verbatim, so the page
costs nothing to serve, cannot expose an API key, and shows exactly the output
the numbers in the evaluation describe. A visitor reading the page and a reader
checking ``docs/EVALUATION.md`` are looking at the same artifact.

Claims cite passages, but a reader wants papers: a title they recognise and a
link they can open. That join happens here, once, against the repository, rather
than in the browser or on every request.
"""

from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.application.evaluate_synthesis import SynthesisReport
from bio_research_copilot.application.ingest import utc_now
from bio_research_copilot.domain import Source

SCOPE_NOTE = "Evidence is limited to titles and abstracts of stored papers, not full text."


class DemoSource(BaseModel):
    """One cited paper, as the page displays it."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    url: str
    journal: str | None = None
    year: int | None = None


class DemoClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    chunk_ids: list[str] = Field(description="Passages supporting this claim.")
    paper_ids: list[str] = Field(description="Papers those passages belong to, in order.")


class DemoAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    type: str
    abstained: bool
    abstain_reason: str | None
    claims: list[DemoClaim]
    sources: list[DemoSource]
    passages: dict[str, str] = Field(
        default_factory=dict,
        description="Cited passage text by chunk id, so a reader can check a claim in place.",
    )
    limitations: list[str]


class DemoData(BaseModel):
    """Everything the page needs, in one file."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    benchmark_version: str
    retriever: str
    synthesizer: str
    disclaimer: str
    answers: list[DemoAnswer]


DISCLAIMER = "For research and educational use only. Not medical advice."


def export_demo(
    repository: PaperRepository,
    report: SynthesisReport,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> DemoData:
    """Display-ready answers, joined to paper metadata.

    A cited paper missing from the repository is skipped rather than raised on,
    because a demo that omits one source card is better than a demo that will
    not build. The claim keeps its paper id either way, so nothing silently
    loses its provenance.
    """
    answers: list[DemoAnswer] = []

    for result in report.results:
        claims: list[DemoClaim] = []
        ordered_papers: list[str] = []
        for record in result.claims:
            paper_ids: list[str] = []
            for passage in record.cited:
                if passage.paper_id not in paper_ids:
                    paper_ids.append(passage.paper_id)
                if passage.paper_id not in ordered_papers:
                    ordered_papers.append(passage.paper_id)
            claims.append(
                DemoClaim(
                    text=record.claim,
                    chunk_ids=list(record.evidence_ids),
                    paper_ids=paper_ids,
                )
            )

        sources: list[DemoSource] = []
        for paper_id in ordered_papers:
            paper = repository.get_paper(Source.PUBMED, paper_id)
            if paper is None:
                continue
            sources.append(
                DemoSource(
                    paper_id=paper.source_id,
                    title=paper.title,
                    url=paper.url,
                    journal=paper.journal,
                    year=paper.publication_date.year if paper.publication_date else None,
                )
            )

        answers.append(
            DemoAnswer(
                id=result.id,
                question=result.question,
                type=result.type,
                abstained=result.abstained,
                abstain_reason=result.abstain_reason,
                claims=claims,
                sources=sources,
                passages={
                    passage.chunk_id: passage.text
                    for record in result.claims
                    for passage in record.cited
                },
                limitations=_limitations(result.abstained, len(result.claims), len(sources)),
            )
        )

    return DemoData(
        generated_at=clock(),
        benchmark_version=report.benchmark_version,
        retriever=report.retriever,
        synthesizer=report.synthesizer,
        disclaimer=DISCLAIMER,
        answers=answers,
    )


def _limitations(abstained: bool, claims: int, papers: int) -> list[str]:
    if abstained:
        return ["No answer: the retrieved evidence does not support one.", SCOPE_NOTE]
    notes = [f"{claims} claims drawn from {papers} papers.", SCOPE_NOTE]
    if papers == 1:
        notes.append("All evidence comes from a single paper, so it is not corroborated.")
    return notes
