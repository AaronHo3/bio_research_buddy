"""Tests for turning a question into a PubMed query.

Planning is an improvement on the deterministic translation, not a prerequisite
for answering, so the property that matters most is that every way it can go
wrong degrades to that translation rather than failing the request. A worse
query is recoverable; an exception is not.
"""

import json
from typing import Any

import httpx2
import pytest

from bio_research_copilot.adapters.literature import LiteratureSourceError
from bio_research_copilot.adapters.query_planner import (
    MAX_TERM_CHARS,
    PLANNER_TOOL,
    PLANNER_TOOL_NAME,
    LlmQueryPlanner,
    build_query_planner,
    is_usable_term,
)

QUESTION = "Tell me about MYBPC3"


def plan_response(term: str, concepts: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": PLANNER_TOOL_NAME,
                "input": {"term": term, "concepts": concepts or []},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 80, "output_tokens": 30},
    }


def make(
    payload: dict[str, Any] | None = None, *, status: int = 200, **kwargs: Any
) -> tuple[LlmQueryPlanner, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        body = payload if payload is not None else {"type": "error", "error": {"message": "x"}}
        return httpx2.Response(status, json=body)

    planner = LlmQueryPlanner(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        **kwargs,
    )
    return planner, seen


class TestUsableTerm:
    @pytest.mark.parametrize(
        "term",
        [
            '(MYBPC3 OR "myosin binding protein C")',
            "statins AND (elderly OR aged)",
            "COVID-19 AND transmission",
            "PD-1/PD-L1 blockade",
        ],
    )
    def test_accepts_a_plain_boolean_query(self, term: str) -> None:
        assert is_usable_term(term)

    @pytest.mark.parametrize(
        "term",
        [
            "",
            "  ",
            "x",
            "MYBPC3[MeSH Terms]",
            "(unbalanced OR parens",
            "term with <script> in it",
        ],
    )
    def test_rejects_what_would_search_worse_than_the_fallback(self, term: str) -> None:
        assert not is_usable_term(term)

    def test_rejects_an_overlong_term(self) -> None:
        assert not is_usable_term("gene " * MAX_TERM_CHARS)


class TestPlanning:
    def test_returns_the_planned_query(self) -> None:
        planner, _ = make(plan_response('(MYBPC3 OR "myosin binding protein C")', ["MYBPC3"]))
        plan = planner.plan(QUESTION)
        assert plan is not None
        assert plan.term == '(MYBPC3 OR "myosin binding protein C")'
        assert plan.concepts == ["MYBPC3"]

    def test_records_which_model_planned(self) -> None:
        planner, _ = make(plan_response("MYBPC3"), model="claude-haiku-4-5-20251001")
        plan = planner.plan(QUESTION)
        assert plan is not None
        assert plan.planner == "anthropic:claude-haiku-4-5-20251001"

    def test_sends_the_question_unchanged(self) -> None:
        """The model needs the phrasing; stripping it first defeats the purpose."""
        planner, seen = make(plan_response("MYBPC3"))
        planner.plan(QUESTION)
        assert QUESTION in seen[0]["messages"][0]["content"]

    def test_forces_the_query_tool(self) -> None:
        planner, seen = make(plan_response("MYBPC3"))
        planner.plan(QUESTION)
        assert seen[0]["tool_choice"] == {"type": "tool", "name": PLANNER_TOOL_NAME}

    def test_records_usage_so_planning_can_be_charged(self) -> None:
        planner, _ = make(plan_response("MYBPC3"))
        planner.plan(QUESTION)
        assert planner.last_usage is not None
        assert planner.last_usage.input_tokens == 80

    def test_the_tool_requires_both_fields(self) -> None:
        assert set(PLANNER_TOOL["input_schema"]["required"]) == {"term", "concepts"}


class TestDegradation:
    """Every failure must fall back to the deterministic translation."""

    def test_a_provider_failure_returns_none(self) -> None:
        planner, _ = make(status=400)
        assert planner.plan(QUESTION) is None

    def test_an_unusable_term_returns_none(self) -> None:
        planner, _ = make(plan_response("MYBPC3[MeSH Terms]"))
        assert planner.plan(QUESTION) is None

    def test_an_empty_term_returns_none(self) -> None:
        planner, _ = make(plan_response(""))
        assert planner.plan(QUESTION) is None

    def test_a_reply_without_a_tool_call_returns_none(self) -> None:
        body = plan_response("MYBPC3")
        body["content"] = [{"type": "text", "text": "I would search for MYBPC3."}]
        planner, _ = make(body)
        assert planner.plan(QUESTION) is None

    def test_missing_api_key_is_reported_at_construction(self) -> None:
        with pytest.raises(LiteratureSourceError, match="needs an API key"):
            build_query_planner("claude-haiku-4-5-20251001")
