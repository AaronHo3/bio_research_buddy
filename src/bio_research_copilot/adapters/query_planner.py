"""Turning a natural-language question into a PubMed query.

A question is not a query. PubMed's search ANDs every term it does not
recognise as a field, so "Tell me about MYBPC3" asks for papers containing
*tell* and *me* and *about* and *MYBPC3*, and matches nothing, while "MYBPC3"
matches several hundred. Stripping known filler words fixes the phrasings
someone thought of and fails on the rest, which is not a foundation for a system
whose entire input is whatever a stranger types.

So the question is planned into a query by a model that knows the vocabulary.
That fixes filler words as a side effect and addresses the harder problem
underneath: a reader asks why their heart muscle thickens, and the papers say
*hypertrophic cardiomyopathy*. No amount of word removal bridges that; supplying
the synonym does.

The plan is a plain boolean query of OR groups joined by AND: alternatives for
one concept inside a group, distinct concepts across them. Field tags such as
``[MeSH Terms]`` are deliberately not used. They are precise when correct and
silently match nothing when wrong, and a demo cannot tell the difference.

This is a cost and a dependency, so the deterministic translation remains the
fallback for when there is no key, no budget, or no usable plan. A worse query
is recoverable; no query at all is not.
"""

import re
import time
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.literature import LiteratureSourceError
from bio_research_copilot.synthesis.llm import (
    DEFAULT_MAX_RETRIES,
    MISSING_KEY,
    Usage,
    _load_sdk,
)

DEFAULT_PLANNER_MODEL = "claude-haiku-4-5-20251001"
"""Planning is a short, well-specified transformation, so the cheap fast model
is the right one: it keeps the added latency near a second and the added cost
negligible against synthesis."""

DEFAULT_PLANNER_MAX_TOKENS = 400

PLANNER_TOOL_NAME = "record_query"

MAX_TERM_CHARS = 400
_ALLOWED_TERM = re.compile(r'^[\w\s()"\'\-,./:+*]+$', re.UNICODE)

PLANNER_SYSTEM_PROMPT = """\
You turn a biomedical question into a PubMed search query.

PubMed combines terms with AND by default, so any word that is not part of the \
subject removes papers. Write a query containing only subject matter: no words \
addressed to an assistant, no verbs describing the request, no punctuation from \
the original sentence.

Build the query from concepts. Group alternative names for one concept with OR \
inside parentheses, and join different concepts with AND. Include the synonyms a \
biomedical author would actually use: gene symbols alongside protein names, \
formal disease names alongside lay descriptions, common abbreviations alongside \
what they stand for. This is the part that matters most, because a reader rarely \
uses the vocabulary the literature uses.

Prefer two or three concepts. Every extra AND group narrows the result set, and \
an over-specified query that returns nothing is worse than a broad one that \
returns too much, because the ranking stage can discard extra papers and cannot \
invent missing ones.

Use no field tags, no date filters and no publication-type filters. Write plain \
terms, parentheses, AND, OR and quoted phrases only.

Examples:
- "Tell me about MYBPC3" becomes (MYBPC3 OR "myosin binding protein C")
- "why does my heart muscle thicken" becomes ("hypertrophic cardiomyopathy" OR \
"cardiac hypertrophy" OR "myocardial hypertrophy")
- "do statins hurt older people" becomes (statins OR "HMG-CoA reductase \
inhibitors") AND (elderly OR aged OR "older adults") AND ("adverse effects" OR \
"side effects" OR safety)\
"""

PLANNER_TOOL: dict[str, Any] = {
    "name": PLANNER_TOOL_NAME,
    "description": "Record the PubMed query for the user's question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": (
                    "The PubMed query: OR groups in parentheses, joined by AND."
                    " Subject matter only, no field tags."
                ),
            },
            "concepts": {
                "type": "array",
                "description": "The distinct concepts the query is built from.",
                "items": {"type": "string"},
            },
        },
        "required": ["term", "concepts"],
    },
}


class SearchPlan(BaseModel):
    """A query and the concepts behind it, so a reader can see what was searched."""

    model_config = ConfigDict(frozen=True)

    term: str = Field(min_length=1)
    concepts: list[str] = Field(default_factory=list)
    planner: str = Field(min_length=1, description="What produced this plan.")


class QueryPlanner(Protocol):
    """Anything that can turn a question into a PubMed query."""

    name: str

    def plan(self, question: str) -> SearchPlan | None:
        """The query, or ``None`` when no usable plan could be produced.

        Returning ``None`` rather than raising keeps planning optional: the
        caller falls back to the deterministic translation and still answers.
        """
        ...


def is_usable_term(term: str) -> bool:
    """Whether a planned query is safe and plausible enough to send.

    A model asked for a query occasionally returns a sentence, an empty string,
    or something carrying field tags it was told not to use. None of those are
    dangerous, but all produce worse results than the deterministic fallback, so
    they are rejected rather than sent.
    """
    cleaned = term.strip()
    if not 2 <= len(cleaned) <= MAX_TERM_CHARS:
        return False
    if "[" in cleaned or "]" in cleaned:
        return False
    if cleaned.count("(") != cleaned.count(")"):
        return False
    return bool(_ALLOWED_TERM.match(cleaned))


class LlmQueryPlanner:
    """Plans a PubMed query with a hosted model.

    Args:
        api_key: Anthropic API key.
        model: Planning model. A small one is appropriate.
        http_client: Injected transport for tests, an ``httpx2.Client``.
        timer: Monotonic clock, injected so latency is assertable.

    Raises:
        SynthesisError: the SDK is missing or no API key was given.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = DEFAULT_PLANNER_MODEL,
        max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
        http_client: Any | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not api_key:
            raise LiteratureSourceError(MISSING_KEY)
        sdk = _load_sdk()
        self.name = f"anthropic:{model}"
        self._model = model
        self._max_tokens = max_tokens
        self._timer = timer
        self._api_error: type[Exception] = sdk.AnthropicError
        self._client = sdk.Anthropic(
            api_key=api_key, http_client=http_client, max_retries=max_retries
        )
        self._last_usage: Usage | None = None

    @property
    def last_usage(self) -> Usage | None:
        """What the last plan cost, so it can be charged against a budget."""
        return self._last_usage

    def plan(self, question: str) -> SearchPlan | None:
        """A query for ``question``, or ``None`` if planning did not work.

        Never raises for a provider failure. Planning is an improvement on the
        deterministic path, not a prerequisite, so a failure here degrades the
        query rather than the answer.
        """
        started = self._timer()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=PLANNER_SYSTEM_PROMPT,
                tools=[PLANNER_TOOL],
                tool_choice={"type": "tool", "name": PLANNER_TOOL_NAME},
                messages=[{"role": "user", "content": f"Question: {question}"}],
            )
        except self._api_error:
            return None
        elapsed = self._timer() - started

        usage = getattr(message, "usage", None)
        self._last_usage = Usage(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            latency_seconds=elapsed,
        )

        payload = _extract_plan(message)
        if payload is None:
            return None
        term = str(payload.get("term") or "").strip()
        if not is_usable_term(term):
            return None
        concepts = [str(c) for c in payload.get("concepts") or [] if str(c).strip()]
        return SearchPlan(term=term, concepts=concepts, planner=self.name)


def build_query_planner(model: str, *, api_key: str | None = None) -> LlmQueryPlanner:
    """Factory in the shape the composition root wires other adapters with."""
    return LlmQueryPlanner(api_key=api_key, model=model)


def _extract_plan(message: Any) -> dict[str, Any] | None:
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != PLANNER_TOOL_NAME:
            continue
        payload = getattr(block, "input", None)
        if isinstance(payload, dict):
            return payload
    return None
