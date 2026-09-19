"""Claude behind the synthesis port.

The extractive provider could not fabricate because it only ever copied. This
one may restate and combine passages, which is precisely the capability it
exists to add, so faithfulness stops being a property of construction and
becomes something that has to be measured.

Two defences stand behind that. The model is handed the real ``chunk_id`` of
every passage and asked to cite those ids; anything it returns that was not
supplied is dropped here and counted, so a fabricated citation degrades an
answer rather than corrupting it. Whatever survives must still pass
``ResearchAnswer``'s validator, which cannot construct an answer citing evidence
it does not hold. This module is best-effort cleanup. That type is the
guarantee, and the two are deliberately not the same layer.

Output is not reproducible. The current SDK exposes no temperature control, so
the same question can be worded differently on two runs. Determinism therefore
stops at the prompt, which is a pure function and golden-tested as one; the
model's wording is not, and that is recorded as a limitation rather than
papered over.

The SDK is an optional dependency, so its import is deferred until construction
and a missing package becomes a typed error rather than a traceback at startup.
It also ships its own vendored HTTP library, which is why an injected client is
typed loosely here: the rest of the project speaks ``httpx`` and this one part
cannot.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bio_research_copilot.domain import Citation, RetrievedEvidence
from bio_research_copilot.synthesis.base import SynthesisError
from bio_research_copilot.synthesis.prompt import (
    CLAIM_TOOL,
    SYSTEM_PROMPT,
    TOOL_NAME,
    build_user_prompt,
)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 2048
"""Ceiling on the reply.

Claims that combine several passages are longer than claims that cite one, and a
reply truncated mid-JSON carries no claims at all rather than fewer. Measured on
live full-text evidence, six combined claims run to roughly 1,200 output tokens,
so 1,024 was under the real requirement. Output tokens are billed as generated,
so headroom here costs nothing until it is used."""
DEFAULT_MAX_CLAIMS = 6
DEFAULT_MAX_RETRIES = 2

# How many times a claims payload may be unwrapped before it is rejected. See
# ``_unwrap_claims``: the observed worst case needs four.
_MAX_UNWRAP = 6

INSTALL_HINT = (
    "LLM synthesis needs the anthropic package, which is an optional dependency;"
    ' install it with "uv sync --extra llm"'
)
MISSING_KEY = (
    "LLM synthesis needs an API key. Set ANTHROPIC_API_KEY in your environment"
    " or .env file, or use the default extractive synthesis instead."
)


@dataclass(frozen=True, slots=True)
class Usage:
    """What the last call cost, and how well the provider behaved.

    ``unsupported_citations`` sits alongside the token counts because it is the
    same kind of fact: something observed about one call. It is the raw material
    for a fabrication rate, which is the number that says whether this provider
    can be trusted with paraphrase at all.
    """

    input_tokens: int
    output_tokens: int
    latency_seconds: float
    unsupported_citations: int = 0


def _load_sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise SynthesisError(INSTALL_HINT) from exc
    return anthropic


class _TruncatedError(SynthesisError):
    """Internal signal: the reply hit the token ceiling mid-structure.

    Distinguished from other malformed output because the remedy differs. A
    truncated reply will truncate again on an identical retry, so the retry has
    to ask for less.
    """


class LlmSynthesizer:
    """Synthesis by a hosted language model, constrained to supplied evidence.

    Args:
        api_key: Anthropic API key. Required; the caller decides where it comes
            from, so this class never reads the environment itself.
        model: Model identifier, recorded on every answer through ``name``.
        max_tokens: Ceiling on the response. Claims are short, so this bounds
            cost without truncating realistic output.
        max_claims: How many claims to keep. Also stated in the prompt, but
            enforced here because a prompt is a request, not a constraint.
        http_client: Injected transport, an ``httpx2.Client`` because the SDK
            vendors that rather than ``httpx``. Tests pass one backed by a mock
            so real request and response shapes are exercised without a network.
        max_retries: Attempts the SDK makes on a retryable status.
        timer: Monotonic clock, injected so latency is assertable.

    Raises:
        SynthesisError: the SDK is not installed, or no API key was given.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        http_client: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not api_key:
            raise SynthesisError(MISSING_KEY)
        if max_claims < 1:
            msg = "max_claims must be at least 1"
            raise ValueError(msg)
        if max_tokens < 1:
            msg = "max_tokens must be at least 1"
            raise ValueError(msg)

        sdk = _load_sdk()
        self.name = f"anthropic:{model}"
        self._model = model
        self._max_tokens = max_tokens
        self._max_claims = max_claims
        self._timer = timer
        self._api_error: type[Exception] = sdk.AnthropicError
        self._client = sdk.Anthropic(
            api_key=api_key, http_client=http_client, max_retries=max_retries
        )
        self._last_usage: Usage | None = None

    @property
    def last_usage(self) -> Usage | None:
        """Cost and behaviour of the most recent call, or ``None`` before one.

        Deliberately per-instance state rather than a return value, so the
        ``SynthesisProvider`` port stays free of provider-specific concepts. One
        instance is therefore not safe to share across threads.
        """
        return self._last_usage

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        """Claims grounded in ``evidence``, best first.

        Raises:
            SynthesisError: the provider call failed, or returned output that
                did not match the required schema.
        """
        if not evidence:
            return []

        empty = Usage(input_tokens=0, output_tokens=0, latency_seconds=0.0)
        try:
            citations, spent = self._attempt(question, evidence, self._max_claims, empty)
        except _TruncatedError:
            # The reply did not fit. Asking for fewer claims is the only retry
            # that can succeed, and a shorter answer beats no answer. The failed
            # attempt's tokens are read back from ``_last_usage``, which the
            # attempt records before raising, so the retry adds to them rather
            # than restarting the count.
            fewer = max(1, self._max_claims // 2)
            citations, spent = self._attempt(question, evidence, fewer, self._last_usage or empty)
        except SynthesisError:
            # Other malformed output is sampling noise; an identical retry works.
            citations, spent = self._attempt(
                question, evidence, self._max_claims, self._last_usage or empty
            )
        self._last_usage = spent
        return citations

    def _attempt(
        self,
        question: str,
        evidence: list[RetrievedEvidence],
        max_claims: int,
        spent: Usage,
    ) -> tuple[list[Citation], Usage]:
        """One call. Usage accumulates across attempts so a retry is still billed."""
        prompt = build_user_prompt(question, evidence, max_claims=max_claims)
        started = self._timer()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                tools=[CLAIM_TOOL],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                messages=[{"role": "user", "content": prompt}],
            )
        except self._api_error as exc:
            msg = f"the language model provider failed: {exc}"
            raise SynthesisError(msg) from exc
        elapsed = self._timer() - started

        usage = getattr(message, "usage", None)
        running = Usage(
            input_tokens=spent.input_tokens + int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=spent.output_tokens + int(getattr(usage, "output_tokens", 0) or 0),
            latency_seconds=spent.latency_seconds + elapsed,
            unsupported_citations=spent.unsupported_citations,
        )
        # Charged even when the attempt fails: the tokens were spent either way,
        # and a budget that ignored failed calls would understate the cost.
        self._last_usage = running

        if getattr(message, "stop_reason", None) == "max_tokens":
            msg = (
                f"the model's reply was cut off at the {self._max_tokens} token limit"
                " before it finished recording claims"
            )
            raise _TruncatedError(msg)

        citations, unsupported = self._ground(_extract_claims(message), evidence)
        return citations, Usage(
            input_tokens=running.input_tokens,
            output_tokens=running.output_tokens,
            latency_seconds=running.latency_seconds,
            unsupported_citations=running.unsupported_citations + unsupported,
        )

    def _ground(
        self, entries: list[dict[str, Any]], evidence: list[RetrievedEvidence]
    ) -> tuple[list[Citation], int]:
        """Keep only claims that cite supplied evidence, counting what was not."""
        position = {item.passage.chunk_id: index for index, item in enumerate(evidence)}
        citations: list[Citation] = []
        seen: set[str] = set()
        unsupported = 0

        for entry in entries:
            claim = str(entry.get("claim") or "").strip()
            raw_ids = entry.get("evidence_ids")
            if not claim or not isinstance(raw_ids, list):
                continue

            supported: list[str] = []
            for ref in raw_ids:
                if isinstance(ref, str) and ref in position:
                    if ref not in supported:
                        supported.append(ref)
                else:
                    unsupported += 1
            if not supported:
                continue

            key = claim.casefold()
            if key in seen:
                continue
            seen.add(key)
            # Cite in retrieval order, so the strongest passage is named first.
            supported.sort(key=lambda ref: position[ref])
            citations.append(Citation(claim=claim, evidence_ids=supported))
            if len(citations) == self._max_claims:
                break

        return citations, unsupported


def build_llm_synthesizer(model: str, *, api_key: str | None = None) -> LlmSynthesizer:
    """Factory in the shape the composition root wires other adapters with.

    The key is a keyword argument the root binds, so the resulting callable takes
    only a model name, like the embedding and reranker factories beside it.
    """
    return LlmSynthesizer(api_key=api_key, model=model)


def _unwrap_claims(value: Any) -> list[Any] | None:
    """The claims array, tolerating a payload that re-encoded it as a string.

    Observed from a real model when abstaining: the tool input arrived as
    ``{"claims": "{\\"claims\\": []}"}``, the array serialised back to JSON and
    wrapped a second time. The API accepts that because the outer shape still
    matches, so the adapter has to. Refusing it would turn a correct abstention
    into an error, which is the opposite of the behaviour the product wants.

    Returns ``None`` when the value is not a claims array under any unwrapping,
    which the caller treats as a broken contract.
    """
    for _ in range(_MAX_UNWRAP):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
            continue
        if isinstance(value, dict) and "claims" in value:
            value = value["claims"]
            continue
        return None
    return None


def _extract_claims(message: Any) -> list[dict[str, Any]]:
    """The tool call's claim list, or an error if the model did not make one.

    ``tool_choice`` forces the call, so a response without it means the provider
    behaved outside its contract and the caller should abstain rather than guess.
    """
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != TOOL_NAME:
            continue
        claims = _unwrap_claims(getattr(block, "input", None))
        if claims is None:
            break
        return [entry for entry in claims if isinstance(entry, dict)]

    msg = "the model did not return claims in the required format"
    raise SynthesisError(msg)
