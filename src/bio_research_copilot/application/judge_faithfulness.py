"""Score a synthesis report for faithfulness, and sample it for human labelling.

Two jobs, kept in one module because they must agree on how a claim is
identified. The automatic judge produces a verdict for every claim; the
worksheet draws a random sample of the same claims for a person to label by
hand. Agreement between the two is only computable because both sides key on the
same ``question_id:claim_index`` pair.

The worksheet needs no API key and no network. That is deliberate: labelling can
start before any judging is paid for, and a person's labels stay independent of
the machine's, which is the whole point of measuring agreement.
"""

import random
from collections.abc import Callable
from datetime import datetime
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.application.evaluate_synthesis import SynthesisReport
from bio_research_copilot.application.ingest import utc_now
from bio_research_copilot.synthesis.base import SynthesisError
from bio_research_copilot.synthesis.faithfulness import Judgement, Verdict

STRICT_VERDICTS: frozenset[str] = frozenset(("supported",))
LENIENT_VERDICTS: frozenset[str] = frozenset(("supported", "partially_supported"))


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str = Field(description='Stable "question_id:claim_index" identifier.')
    question_id: str
    question_type: str
    claim: str
    evidence_ids: list[str]
    verdict: Verdict
    reason: str


class FaithfulnessReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesizer: str
    judge: str
    generated_at: datetime
    claims: int
    supported: int
    partially_supported: int
    unsupported: int
    strict_rate: float = Field(description="Fully supported claims, as a fraction of all claims.")
    lenient_rate: float = Field(description="Supported or partially supported.")
    input_tokens: int
    output_tokens: int
    unjudged: int = Field(
        default=0,
        description="Claims the judge could not rule on. Excluded from the rates.",
    )
    by_type: dict[str, float] = Field(description="Strict rate per question type.")
    verdicts: list[ClaimVerdict]


class WorksheetItem(BaseModel):
    """One claim for a person to label, with everything needed to decide."""

    model_config = ConfigDict(frozen=False)

    key: str
    claim: str
    passages: list[str]
    verdict: str = Field(
        default="",
        description="Fill in: supported, partially_supported, or unsupported.",
    )
    note: str = ""


class Worksheet(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesizer: str
    seed: int
    population: int = Field(description="Claims the sample was drawn from.")
    instructions: list[str]
    items: list[WorksheetItem]


WORKSHEET_INSTRUCTIONS = [
    "For each item, read the passages and then the claim.",
    "Ask only: could a careful reader derive this claim from these passages alone?",
    "Do not judge whether the claim is true in general, or whether it answers a question.",
    "supported: every part of the claim follows from the passages.",
    "partially_supported: the passages support part of it, the claim goes further.",
    "unsupported: the passages do not support the claim, or contradict it.",
    "Label every item before comparing. Do not look at the machine verdicts first.",
]


class ClaimRow(NamedTuple):
    """One claim flattened out of the report, with everything judging needs."""

    key: str
    question_id: str
    question_type: str
    claim: str
    evidence_ids: list[str]
    passages: list[str]


def iter_claims(report: SynthesisReport) -> list[ClaimRow]:
    """Every claim in the report, in report order, keyed for later joining."""
    return [
        ClaimRow(
            key=f"{result.id}:{index}",
            question_id=result.id,
            question_type=result.type,
            claim=record.claim,
            evidence_ids=list(record.evidence_ids),
            passages=[passage.text for passage in record.cited],
        )
        for result in report.results
        for index, record in enumerate(result.claims)
    ]


def build_worksheet(report: SynthesisReport, *, size: int, seed: int) -> Worksheet:
    """A random sample of claims for a person to label.

    Args:
        size: How many claims to draw. The whole population is used when the
            report holds fewer.
        seed: Fixed so the same sample can be regenerated and checked.

    Raises:
        ValueError: ``size`` is not positive, or the report has no claims.
    """
    if size < 1:
        msg = "size must be at least 1"
        raise ValueError(msg)
    rows = iter_claims(report)
    if not rows:
        msg = "the report contains no claims to sample"
        raise ValueError(msg)

    drawn = random.Random(seed).sample(rows, min(size, len(rows)))
    drawn.sort(key=lambda row: row.key)
    return Worksheet(
        synthesizer=report.synthesizer,
        seed=seed,
        population=len(rows),
        instructions=list(WORKSHEET_INSTRUCTIONS),
        items=[WorksheetItem(key=row.key, claim=row.claim, passages=row.passages) for row in drawn],
    )


JudgeFn = Callable[[str, list[str]], Judgement]


def judge_report(
    report: SynthesisReport,
    judge: JudgeFn,
    judge_name: str,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> FaithfulnessReport:
    """Judge every claim in a synthesis report.

    A claim the judge cannot rule on, even after its own retry, is counted as
    unjudged and left out of the rates. It is not given a verdict: guessing
    would corrupt the number, and discarding the whole run for one flaky call
    would throw away thirty good verdicts to punish one bad one. The count is
    reported so a reader can see how much was actually measured.

    Raises:
        ValueError: the report contains no claims, or none could be judged.
    """
    rows = iter_claims(report)
    if not rows:
        msg = "the report contains no claims to judge"
        raise ValueError(msg)

    verdicts: list[ClaimVerdict] = []
    input_tokens = 0
    output_tokens = 0
    unjudged = 0
    for row in rows:
        try:
            result = judge(row.claim, row.passages)
        except SynthesisError:
            unjudged += 1
            continue
        input_tokens += result.input_tokens
        output_tokens += result.output_tokens
        verdicts.append(
            ClaimVerdict(
                key=row.key,
                question_id=row.question_id,
                question_type=row.question_type,
                claim=row.claim,
                evidence_ids=row.evidence_ids,
                verdict=result.verdict,
                reason=result.reason,
            )
        )

    if not verdicts:
        msg = f"the judge could not rule on any of the {len(rows)} claims"
        raise ValueError(msg)

    counts = {name: sum(1 for v in verdicts if v.verdict == name) for name in _VERDICT_NAMES}
    total = len(verdicts)
    by_type: dict[str, float] = {}
    for qtype in sorted({v.question_type for v in verdicts}):
        group = [v for v in verdicts if v.question_type == qtype]
        by_type[qtype] = sum(1 for v in group if v.verdict in STRICT_VERDICTS) / len(group)

    return FaithfulnessReport(
        synthesizer=report.synthesizer,
        judge=judge_name,
        generated_at=clock(),
        claims=total,
        supported=counts["supported"],
        partially_supported=counts["partially_supported"],
        unsupported=counts["unsupported"],
        strict_rate=sum(1 for v in verdicts if v.verdict in STRICT_VERDICTS) / total,
        lenient_rate=sum(1 for v in verdicts if v.verdict in LENIENT_VERDICTS) / total,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        unjudged=unjudged,
        by_type=by_type,
        verdicts=verdicts,
    )


_VERDICT_NAMES = ("supported", "partially_supported", "unsupported")
