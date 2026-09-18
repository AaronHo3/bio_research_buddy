"""Tests for the faithfulness judge and the human-labelling worksheet.

Two properties matter more than the arithmetic. A judge failure must never be
defaulted to a verdict, because silently treating an error as "supported" would
inflate the headline number. And the worksheet must key on exactly the same
identifier as the judge, because agreement between them is only computable if
the two sides can be joined.
"""

import json
from datetime import UTC, datetime
from typing import Any

import httpx2
import pytest

from bio_research_copilot.application.evaluate_synthesis import (
    AbstentionSummary,
    CitedPassage,
    ClaimRecord,
    SynthesisQuestionResult,
    SynthesisReport,
)
from bio_research_copilot.application.judge_faithfulness import (
    build_worksheet,
    iter_claims,
    judge_report,
)
from bio_research_copilot.synthesis import SynthesisError
from bio_research_copilot.synthesis.faithfulness import (
    JUDGE_TOOL,
    JUDGE_TOOL_NAME,
    Judgement,
    LlmFaithfulnessJudge,
    build_judge,
    build_judge_prompt,
)

T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def claim(text: str, chunk: str, passage: str) -> ClaimRecord:
    return ClaimRecord(
        claim=text,
        evidence_ids=[chunk],
        cited=[
            CitedPassage(
                chunk_id=chunk,
                paper_id=chunk.split(":")[1],
                section="abstract",
                text=passage,
            )
        ],
    )


def result(qid: str, qtype: str, claims: list[ClaimRecord]) -> SynthesisQuestionResult:
    return SynthesisQuestionResult(
        id=qid,
        topic="t",
        type=qtype,
        question="q?",
        abstained=not claims,
        abstain_reason=None,
        claims=claims,
        cited_papers=[],
        relevant=[],
        citation_precision=None,
        unsupported_citations=0,
        input_tokens=0,
        output_tokens=0,
        latency_ms=1.0,
    )


def report(results: list[SynthesisQuestionResult]) -> SynthesisReport:
    return SynthesisReport(
        benchmark_version="v",
        retriever="bm25-v1",
        synthesizer="extractive-v1",
        generated_at=T1,
        questions=len(results),
        abstention=AbstentionSummary(
            answerable=len(results),
            answered=len(results),
            false_abstentions=0,
            unanswerable=0,
            abstained=0,
            false_answers=0,
        ),
        mean_citation_precision=None,
        mean_claims_per_answer=1.0,
        mean_papers_per_answer=1.0,
        unsupported_citations=0,
        total_input_tokens=0,
        total_output_tokens=0,
        mean_latency_ms=1.0,
        by_type={},
        results=results,
    )


SAMPLE = report(
    [
        result(
            "q1",
            "mechanism",
            [claim("A.", "pubmed:1:0", "Text one."), claim("B.", "pubmed:2:0", "Text two.")],
        ),
        result("q2", "comparison", [claim("C.", "pubmed:3:0", "Text three.")]),
    ]
)


def verdict_response(verdict: str, reason: str = "because") -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": JUDGE_TOOL_NAME,
                "input": {"verdict": verdict, "reason": reason},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 60, "output_tokens": 15},
    }


def make(
    payload: dict[str, Any] | None = None, *, status: int = 200, **kwargs: Any
) -> tuple[LlmFaithfulnessJudge, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        body = payload if payload is not None else {"type": "error", "error": {"message": "bad"}}
        return httpx2.Response(status, json=body)

    judge = LlmFaithfulnessJudge(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        **kwargs,
    )
    return judge, seen


class TestJudgePrompt:
    def test_shows_the_passages_and_the_claim(self) -> None:
        built = build_judge_prompt("A claim.", ["Passage one.", "Passage two."])
        assert "Passage one." in built
        assert "Passage two." in built
        assert "A claim." in built

    def test_evidence_precedes_the_claim(self) -> None:
        built = build_judge_prompt("A claim.", ["Passage one."])
        assert built.index("</evidence>") < built.index("A claim.")

    def test_the_verdict_schema_is_a_closed_set(self) -> None:
        """A free-text verdict could not be counted."""
        enum = JUDGE_TOOL["input_schema"]["properties"]["verdict"]["enum"]
        assert enum == ["supported", "partially_supported", "unsupported"]

    def test_a_reason_is_required(self) -> None:
        assert set(JUDGE_TOOL["input_schema"]["required"]) == {"verdict", "reason"}


class TestJudge:
    def test_returns_the_verdict_and_reason(self) -> None:
        judge, _ = make(verdict_response("partially_supported", "goes further"))
        got = judge.judge("A claim.", ["A passage."])
        assert got.verdict == "partially_supported"
        assert got.reason == "goes further"

    def test_the_question_is_never_shown_to_the_judge(self) -> None:
        """Faithfulness is claim against evidence; relevance is measured elsewhere."""
        judge, seen = make(verdict_response("supported"))
        judge.judge("A claim.", ["A passage."])
        assert "Question:" not in seen[0]["messages"][0]["content"]

    def test_accumulates_usage_across_calls(self) -> None:
        judge, _ = make(verdict_response("supported"))
        judge.judge("A.", ["p"])
        judge.judge("B.", ["p"])
        assert judge.total_usage.input_tokens == 120
        assert judge.total_usage.output_tokens == 30

    def test_a_claim_with_no_passages_cannot_be_judged(self) -> None:
        judge, _ = make(verdict_response("supported"))
        with pytest.raises(SynthesisError, match="no cited passages"):
            judge.judge("A claim.", [])

    def test_a_provider_error_is_not_defaulted_to_a_verdict(self) -> None:
        judge, _ = make(status=400)
        with pytest.raises(SynthesisError, match="judge failed"):
            judge.judge("A claim.", ["A passage."])

    def test_an_unknown_verdict_is_rejected(self) -> None:
        judge, _ = make(verdict_response("probably_fine"))
        with pytest.raises(SynthesisError, match="required format"):
            judge.judge("A claim.", ["A passage."])

    def test_a_reencoded_payload_is_unwrapped(self) -> None:
        """Same live-API quirk the claims adapter already handles."""
        body = verdict_response("supported")
        body["content"] = [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": JUDGE_TOOL_NAME,
                "input": '{"verdict": "supported", "reason": "ok"}',
            }
        ]
        judge, _ = make(body)
        assert judge.judge("A claim.", ["A passage."]).verdict == "supported"

    def test_missing_api_key_is_reported(self) -> None:
        with pytest.raises(SynthesisError, match="needs an API key"):
            build_judge("claude-sonnet-5")


class TestJudgeReport:
    def test_counts_every_verdict_kind(self) -> None:
        verdicts = iter(["supported", "partially_supported", "unsupported"])
        scored = judge_report(
            SAMPLE,
            lambda c, p: Judgement(verdict=next(verdicts), reason="r"),  # type: ignore[arg-type]
            "fake-judge",
        )
        assert (scored.supported, scored.partially_supported, scored.unsupported) == (1, 1, 1)
        assert scored.claims == 3

    def test_strict_and_lenient_rates_differ_on_partials(self) -> None:
        verdicts = iter(["supported", "partially_supported", "unsupported"])
        scored = judge_report(
            SAMPLE,
            lambda c, p: Judgement(verdict=next(verdicts), reason="r"),  # type: ignore[arg-type]
            "fake-judge",
        )
        assert scored.strict_rate == pytest.approx(1 / 3)
        assert scored.lenient_rate == pytest.approx(2 / 3)

    def test_judges_each_claim_against_its_own_passages(self) -> None:
        seen: list[tuple[str, list[str]]] = []

        def judge(claim_text: str, passages: list[str]) -> Judgement:
            seen.append((claim_text, passages))
            return Judgement(verdict="supported", reason="r")

        judge_report(SAMPLE, judge, "fake-judge")
        assert seen == [("A.", ["Text one."]), ("B.", ["Text two."]), ("C.", ["Text three."])]

    def test_groups_the_strict_rate_by_question_type(self) -> None:
        verdicts = iter(["supported", "unsupported", "supported"])
        scored = judge_report(
            SAMPLE,
            lambda c, p: Judgement(verdict=next(verdicts), reason="r"),  # type: ignore[arg-type]
            "fake-judge",
        )
        assert scored.by_type["mechanism"] == pytest.approx(0.5)
        assert scored.by_type["comparison"] == pytest.approx(1.0)

    def test_a_report_with_no_claims_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no claims to judge"):
            judge_report(report([result("q1", "mechanism", [])]), lambda c, p: None, "j")  # type: ignore[arg-type,return-value]


class TestWorksheet:
    def test_draws_the_requested_number(self) -> None:
        sheet = build_worksheet(SAMPLE, size=2, seed=1)
        assert len(sheet.items) == 2
        assert sheet.population == 3

    def test_the_same_seed_draws_the_same_sample(self) -> None:
        first = build_worksheet(SAMPLE, size=2, seed=7)
        second = build_worksheet(SAMPLE, size=2, seed=7)
        assert [i.key for i in first.items] == [i.key for i in second.items]

    def test_a_size_beyond_the_population_takes_everything(self) -> None:
        assert len(build_worksheet(SAMPLE, size=99, seed=1).items) == 3

    def test_items_carry_their_passages_so_labelling_needs_nothing_else(self) -> None:
        for item in build_worksheet(SAMPLE, size=3, seed=1).items:
            assert item.claim
            assert item.passages

    def test_verdict_starts_blank_for_a_person_to_fill(self) -> None:
        assert all(item.verdict == "" for item in build_worksheet(SAMPLE, size=3, seed=1).items)

    def test_keys_match_the_judge_so_the_two_can_be_joined(self) -> None:
        """Agreement is only computable if both sides key identically."""
        sheet_keys = {item.key for item in build_worksheet(SAMPLE, size=3, seed=1).items}
        judged = judge_report(
            SAMPLE, lambda c, p: Judgement(verdict="supported", reason="r"), "fake-judge"
        )
        assert sheet_keys == {v.key for v in judged.verdicts}

    def test_carries_the_instructions_a_labeller_needs(self) -> None:
        sheet = build_worksheet(SAMPLE, size=1, seed=1)
        assert any("partially_supported" in line for line in sheet.instructions)

    def test_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="size must be at least 1"):
            build_worksheet(SAMPLE, size=0, seed=1)

    def test_a_report_with_no_claims_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no claims to sample"):
            build_worksheet(report([result("q1", "mechanism", [])]), size=1, seed=1)


class TestIterClaims:
    def test_flattens_in_report_order_with_stable_keys(self) -> None:
        assert [row.key for row in iter_claims(SAMPLE)] == ["q1:0", "q1:1", "q2:0"]
