"""Judging whether a claim follows from the passage it cites.

This is the one thing the automatic evaluation cannot do. Citation correctness
asks whether the cited paper was relevant to the question; faithfulness asks
something narrower and harder: granted this passage, does this claim follow from
it? A claim can cite an entirely relevant paper and still say more than that
paper said.

The judge is shown the claim and the passages it cites, and deliberately not the
question. Faithfulness is a relation between a claim and its evidence alone.
Showing the question would invite the judge to reward a claim for being a good
answer, which is a different property already measured elsewhere.

Three verdicts rather than two, because the interesting failure is rarely a flat
contradiction. It is a claim that generalises past its evidence: true of the
study described, asserted of the field. ``partially_supported`` is where that
lands, and collapsing it into either neighbour would hide the most common way
grounded synthesis actually goes wrong.

Using a model to grade a model is a real weakness, not a detail. The remedy is
not to claim otherwise but to measure it: every verdict here is checkable against
a human label, and the evaluation reports agreement rather than assuming it.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from bio_research_copilot.synthesis.base import SynthesisError
from bio_research_copilot.synthesis.llm import (
    DEFAULT_MAX_RETRIES,
    MISSING_KEY,
    Usage,
    _load_sdk,
)

Verdict = Literal["supported", "partially_supported", "unsupported"]

DEFAULT_JUDGE_MODEL = "claude-sonnet-5"
DEFAULT_JUDGE_MAX_TOKENS = 512

JUDGE_TOOL_NAME = "record_verdict"

JUDGE_SYSTEM_PROMPT = """\
You judge whether a claim is supported by the evidence passages quoted beneath \
it. You are not judging whether the claim is true in general, whether it is \
useful, or whether it answers any particular question. You are judging one \
thing: could a careful reader derive this claim from these passages alone?

Verdicts:

- supported: every part of the claim follows from the passages. A faithful \
paraphrase counts as supported; identical wording is not required.
- partially_supported: the passages support part of the claim but the claim \
goes further than they do. This covers a claim that generalises a single study \
to a field, states a cause where the passage reports an association, drops a \
hedge the passage carried, or adds a specific the passage does not contain.
- unsupported: the passages do not support the claim, or contradict it.

Judge only against the text supplied. If the claim is correct as biomedical \
fact but the passages do not establish it, that is not supported. Give a brief \
reason quoting or naming the specific wording that decided the verdict.\
"""

JUDGE_TOOL: dict[str, Any] = {
    "name": JUDGE_TOOL_NAME,
    "description": "Record the faithfulness verdict for one claim against its cited passages.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supported", "partially_supported", "unsupported"],
                "description": "Whether the claim follows from the passages alone.",
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences naming the wording that decided it.",
            },
        },
        "required": ["verdict", "reason"],
    },
}


@dataclass(frozen=True, slots=True)
class Judgement:
    """One verdict, with what it cost to obtain."""

    verdict: Verdict
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0


class FaithfulnessJudge(Protocol):
    """The port a judge implements, so the evaluation can be driven by a fake."""

    name: str

    def judge(self, claim: str, passages: list[str]) -> Judgement:
        """The verdict for one claim against the passages it cites."""
        ...


def build_judge_prompt(claim: str, passages: list[str]) -> str:
    """The claim and its evidence, evidence first."""
    quoted = "\n".join(f"<passage>\n{text}\n</passage>" for text in passages)
    return (
        f"<evidence>\n{quoted}\n</evidence>\n\n"
        f"Claim: {claim}\n\n"
        "Record whether this claim follows from the passages above."
    )


class LlmFaithfulnessJudge:
    """Faithfulness verdicts from a hosted model.

    Args:
        api_key: Anthropic API key.
        model: Judge model. Recorded on the report, because a verdict is only
            meaningful alongside what produced it.
        http_client: Injected transport for tests, an ``httpx2.Client``.
        timer: Monotonic clock, injected so latency is assertable.

    Raises:
        SynthesisError: the SDK is missing or no API key was given.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
        http_client: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not api_key:
            raise SynthesisError(MISSING_KEY)
        if max_tokens < 1:
            msg = "max_tokens must be at least 1"
            raise ValueError(msg)

        sdk = _load_sdk()
        self.name = f"anthropic:{model}"
        self._model = model
        self._max_tokens = max_tokens
        self._timer = timer
        self._api_error: type[Exception] = sdk.AnthropicError
        self._client = sdk.Anthropic(
            api_key=api_key, http_client=http_client, max_retries=max_retries
        )
        self._total = Usage(input_tokens=0, output_tokens=0, latency_seconds=0.0)

    @property
    def total_usage(self) -> Usage:
        """Everything this judge has spent, accumulated across calls."""
        return self._total

    def judge(self, claim: str, passages: list[str]) -> Judgement:
        """One verdict.

        A malformed reply is retried once. Observed from the live API: a model
        occasionally emits a degenerate tool call such as
        ``{"parameter name": "verdict"}``, which carries no verdict at all. That
        is sampling noise rather than a considered refusal, and a second attempt
        usually succeeds.

        Raises:
            SynthesisError: the call failed, or the verdict was still malformed
                after a retry. A missing verdict is never defaulted to anything:
                treating a failure as "supported" would inflate the rate this
                whole measurement exists to establish.
        """
        if not passages:
            msg = "a claim with no cited passages cannot be judged"
            raise SynthesisError(msg)
        try:
            return self._judge_once(claim, passages)
        except SynthesisError:
            return self._judge_once(claim, passages)

    def _judge_once(self, claim: str, passages: list[str]) -> Judgement:
        started = self._timer()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": JUDGE_TOOL_NAME},
                messages=[{"role": "user", "content": build_judge_prompt(claim, passages)}],
            )
        except self._api_error as exc:
            msg = f"the faithfulness judge failed: {exc}"
            raise SynthesisError(msg) from exc
        elapsed = self._timer() - started

        verdict, reason = _extract_verdict(message)
        usage = getattr(message, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        self._total = Usage(
            input_tokens=self._total.input_tokens + inp,
            output_tokens=self._total.output_tokens + out,
            latency_seconds=self._total.latency_seconds + elapsed,
        )
        return Judgement(
            verdict=verdict,
            reason=reason,
            input_tokens=inp,
            output_tokens=out,
            latency_seconds=elapsed,
        )


def build_judge(model: str, *, api_key: str | None = None) -> LlmFaithfulnessJudge:
    """Factory in the shape the composition root wires other adapters with."""
    return LlmFaithfulnessJudge(api_key=api_key, model=model)


_VERDICTS: frozenset[str] = frozenset(("supported", "partially_supported", "unsupported"))


def _extract_verdict(message: Any) -> tuple[Verdict, str]:
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != JUDGE_TOOL_NAME:
            continue
        payload = _unwrap_payload(getattr(block, "input", None))
        if payload is None:
            break
        verdict = payload.get("verdict")
        if verdict not in _VERDICTS:
            break
        reason = str(payload.get("reason") or "").strip()
        return verdict, reason

    msg = "the judge did not return a verdict in the required format"
    raise SynthesisError(msg)


def _unwrap_payload(value: Any) -> dict[str, Any] | None:
    """Tolerate a re-encoded tool payload, as the claims adapter already must.

    A live model returned its tool input serialised back to JSON rather than as
    an object; see ``llm._unwrap_claims``. The same handling applies here.
    """
    for _ in range(4):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
            continue
        return None
    return None


__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_TOOL",
    "JUDGE_TOOL_NAME",
    "FaithfulnessJudge",
    "Judgement",
    "LlmFaithfulnessJudge",
    "Verdict",
    "build_judge",
    "build_judge_prompt",
]
