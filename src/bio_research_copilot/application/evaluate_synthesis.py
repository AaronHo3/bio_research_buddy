"""Run a synthesis provider over the benchmark and score what can be scored.

Four things are measurable here without a human, and one is not.

Citation *integrity* is not among them. ``ResearchAnswer`` refuses to hold a
citation pointing outside its own evidence, so integrity is one hundred percent
by construction and reporting it would be measuring a tautology. What is worth
counting is the attempt: ``unsupported_citations`` records how often a provider
tried to cite something it was never given. That is a fabrication-attempt rate,
and unlike integrity it can be non-zero.

Citation *correctness* reuses the relevance judgments already made for
retrieval. Of the papers an answer actually cites, how many were judged relevant
to that question? A provider can cite perfectly legal evidence that has nothing
to do with what was asked, and this is the number that catches it.

Abstention is a two-by-two. Answering an unanswerable question and refusing an
answerable one are different failures with different costs, and for a biomedical
tool the first is far worse, so they are counted separately rather than folded
into one accuracy figure.

Cost and latency come from the provider's own usage record.

Faithfulness -- whether a claim actually follows from the passage it cites --
needs judgment and is deliberately absent. This module writes every claim out
with its cited text so that judging can happen against a fixed artifact rather
than against a fresh run that may word things differently.
"""

import time
from collections.abc import Callable
from datetime import datetime
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.persistence import PaperRepository
from bio_research_copilot.application.answer import (
    DEFAULT_EVIDENCE_LIMIT,
    answer_question,
)
from bio_research_copilot.application.ingest import utc_now
from bio_research_copilot.application.query import RetrieverFactory, build_bm25
from bio_research_copilot.domain import ResearchAnswer
from bio_research_copilot.evaluation import Benchmark
from bio_research_copilot.synthesis import DEFAULT_CHAR_BUDGET, SynthesisProvider


class CitedPassage(BaseModel):
    """A passage as it was shown to synthesis, kept so judging needs no database."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    paper_id: str
    section: str
    text: str


class ClaimRecord(BaseModel):
    """One claim and the evidence behind it, in a form a judge can read alone."""

    model_config = ConfigDict(frozen=True)

    claim: str
    evidence_ids: list[str]
    cited: list[CitedPassage]


class SynthesisQuestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    topic: str
    type: str
    question: str
    abstained: bool
    abstain_reason: str | None
    claims: list[ClaimRecord]
    cited_papers: list[str]
    relevant: list[str]
    citation_precision: float | None = Field(
        description="Cited papers that were judged relevant. None when nothing was cited."
    )
    unsupported_citations: int
    input_tokens: int
    output_tokens: int
    latency_ms: float


class AbstentionSummary(BaseModel):
    """The two-by-two. The diagonal is correct behaviour."""

    model_config = ConfigDict(frozen=True)

    answerable: int
    answered: int
    false_abstentions: int = Field(description="Answerable questions refused.")
    unanswerable: int
    abstained: int
    false_answers: int = Field(description="Unanswerable questions answered anyway.")


class SynthesisTypeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: int
    answered: int
    mean_claims: float
    mean_citation_precision: float | None


class SynthesisReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_version: str
    retriever: str
    synthesizer: str
    generated_at: datetime
    questions: int
    abstention: AbstentionSummary
    mean_citation_precision: float | None
    mean_claims_per_answer: float
    mean_papers_per_answer: float
    unsupported_citations: int = Field(description="Fabrication attempts across all questions.")
    total_input_tokens: int
    total_output_tokens: int
    mean_latency_ms: float
    by_type: dict[str, SynthesisTypeSummary]
    results: list[SynthesisQuestionResult]


def _record_claims(answer: ResearchAnswer) -> list[ClaimRecord]:
    passages = {item.passage.chunk_id: item.passage for item in answer.evidence}
    records: list[ClaimRecord] = []
    for citation in answer.citations:
        cited = [
            CitedPassage(chunk_id=p.chunk_id, paper_id=p.source_id, section=p.section, text=p.text)
            for ref in citation.evidence_ids
            if (p := passages.get(ref)) is not None
        ]
        records.append(
            ClaimRecord(claim=citation.claim, evidence_ids=list(citation.evidence_ids), cited=cited)
        )
    return records


def _cited_papers(claims: list[ClaimRecord]) -> list[str]:
    ordered: list[str] = []
    for record in claims:
        for passage in record.cited:
            if passage.paper_id not in ordered:
                ordered.append(passage.paper_id)
    return ordered


def evaluate_synthesis(
    repository: PaperRepository,
    benchmark: Benchmark,
    *,
    build_retriever: RetrieverFactory = build_bm25,
    synthesizer: SynthesisProvider | None = None,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    min_score: float | None = None,
    clock: Callable[[], datetime] = utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> SynthesisReport:
    """Answer every benchmark question and score the answers.

    Raises:
        SynthesisError: the provider failed on a question. Nothing is caught
            here, because a partial report presented as a complete one is worse
            than no report.
    """
    results: list[SynthesisQuestionResult] = []
    retrieval_method = ""
    synthesis_method = ""

    for q in benchmark.questions:
        started = timer()
        answer = answer_question(
            repository,
            q.question,
            build_retriever=build_retriever,
            synthesizer=synthesizer,
            evidence_limit=evidence_limit,
            char_budget=char_budget,
            min_score=min_score,
        )
        latency_ms = (timer() - started) * 1000.0

        retrieval_method = answer.retrieval_method
        synthesis_method = answer.synthesis_method
        claims = _record_claims(answer)
        cited = _cited_papers(claims)
        relevant = set(q.relevant)
        # Undefined rather than zero when nothing was cited: an abstention is not
        # a wrong citation, and averaging a zero in would punish correct refusal.
        precision = (
            len([p for p in cited if p in relevant]) / len(cited) if cited and relevant else None
        )
        usage = getattr(synthesizer, "last_usage", None)

        results.append(
            SynthesisQuestionResult(
                id=q.id,
                topic=q.topic,
                type=q.type,
                question=q.question,
                abstained=answer.abstained,
                abstain_reason=answer.limitations[0] if answer.abstained else None,
                claims=claims,
                cited_papers=cited,
                relevant=list(q.relevant),
                citation_precision=precision,
                unsupported_citations=getattr(usage, "unsupported_citations", 0),
                input_tokens=getattr(usage, "input_tokens", 0),
                output_tokens=getattr(usage, "output_tokens", 0),
                latency_ms=latency_ms,
            )
        )

    answerable = [r for r in results if r.type != "unanswerable"]
    unanswerable = [r for r in results if r.type == "unanswerable"]
    scored = [r.citation_precision for r in results if r.citation_precision is not None]

    by_type: dict[str, SynthesisTypeSummary] = {}
    for qtype in sorted({r.type for r in results}):
        group = [r for r in results if r.type == qtype]
        group_scored = [r.citation_precision for r in group if r.citation_precision is not None]
        by_type[qtype] = SynthesisTypeSummary(
            questions=len(group),
            answered=sum(1 for r in group if not r.abstained),
            mean_claims=mean(len(r.claims) for r in group),
            mean_citation_precision=mean(group_scored) if group_scored else None,
        )

    answered = [r for r in results if not r.abstained]

    return SynthesisReport(
        benchmark_version=benchmark.version,
        retriever=retrieval_method,
        synthesizer=synthesis_method,
        generated_at=clock(),
        questions=len(results),
        abstention=AbstentionSummary(
            answerable=len(answerable),
            answered=sum(1 for r in answerable if not r.abstained),
            false_abstentions=sum(1 for r in answerable if r.abstained),
            unanswerable=len(unanswerable),
            abstained=sum(1 for r in unanswerable if r.abstained),
            false_answers=sum(1 for r in unanswerable if not r.abstained),
        ),
        mean_citation_precision=mean(scored) if scored else None,
        mean_claims_per_answer=mean(len(r.claims) for r in answered) if answered else 0.0,
        mean_papers_per_answer=mean(len(r.cited_papers) for r in answered) if answered else 0.0,
        unsupported_citations=sum(r.unsupported_citations for r in results),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        mean_latency_ms=mean(r.latency_ms for r in results),
        by_type=by_type,
        results=results,
    )
