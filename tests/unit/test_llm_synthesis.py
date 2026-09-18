"""Tests for the hosted-model synthesis provider.

The provider is driven through a mock transport rather than a stub client, so
every test exercises the request the SDK actually serialises and the response
shape it actually parses. A stub would pass even if the tool schema were wrong.

The central property under test is not answer quality, which cannot be asserted
offline, but containment: whatever the model returns, only claims citing
supplied evidence come out, and the rest is counted rather than silently lost.
"""

import hashlib
import json
from typing import Any

import anthropic
import httpx2
import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, Source
from bio_research_copilot.synthesis import SynthesisError
from bio_research_copilot.synthesis.llm import LlmSynthesizer, build_llm_synthesizer
from bio_research_copilot.synthesis.prompt import TOOL_NAME

QUESTION = "What causes off-target editing by base editors?"


def evidence(chunk: str, text: str, rank: int) -> RetrievedEvidence:
    pmid, position = chunk.split(":")
    return RetrievedEvidence(
        passage=Passage(
            source=Source.PUBMED,
            source_id=pmid,
            chunk_id=f"pubmed:{chunk}",
            position=int(position),
            section="abstract",
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            strategy="test",
        ),
        rank=rank,
        score=1.0 / rank,
        method="test",
    )


EVIDENCE = [
    evidence("1:0", "Deaminase activity causes off-target editing.", 1),
    evidence("2:0", "Guide RNA design affects editing specificity.", 2),
]


def tool_response(claims: list[dict[str, Any]], **usage: int) -> dict[str, Any]:
    """A well-formed Messages response carrying a forced tool call."""
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": TOOL_NAME, "input": {"claims": claims}}
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": usage.get("input_tokens", 100), "output_tokens": 50},
    }


def make(
    payload: dict[str, Any] | None = None, *, status: int = 200, **kwargs: Any
) -> tuple[LlmSynthesizer, list[dict[str, Any]]]:
    """A synthesizer wired to a mock transport, plus the captured requests."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        body = payload if payload is not None else {"type": "error", "error": {"message": "bad"}}
        return httpx2.Response(status, json=body)

    provider = LlmSynthesizer(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
        **kwargs,
    )
    return provider, seen


class TestConstruction:
    def test_missing_api_key_is_a_synthesis_error(self) -> None:
        with pytest.raises(SynthesisError, match="needs an API key"):
            LlmSynthesizer(api_key=None)

    def test_empty_api_key_is_a_synthesis_error(self) -> None:
        with pytest.raises(SynthesisError, match="needs an API key"):
            LlmSynthesizer(api_key="")

    def test_name_records_provider_and_model(self) -> None:
        provider, _ = make(tool_response([]), model="claude-haiku-4-5-20251001")
        assert provider.name == "anthropic:claude-haiku-4-5-20251001"

    def test_claim_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_claims must be at least 1"):
            LlmSynthesizer(api_key="k", max_claims=0)

    def test_token_limit_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            LlmSynthesizer(api_key="k", max_tokens=0)

    def test_factory_builds_a_provider_for_the_named_model(self) -> None:
        provider = build_llm_synthesizer("claude-sonnet-5", api_key="k")
        assert provider.name == "anthropic:claude-sonnet-5"

    def test_factory_without_a_key_fails_with_guidance(self) -> None:
        with pytest.raises(SynthesisError, match="extractive"):
            build_llm_synthesizer("claude-sonnet-5")


class TestRequest:
    def test_forces_the_claim_tool_so_prose_cannot_come_back(self) -> None:
        provider, seen = make(tool_response([]))
        provider.synthesize(QUESTION, EVIDENCE)
        assert seen[0]["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
        assert seen[0]["tools"][0]["name"] == TOOL_NAME

    def test_evidence_ids_are_required_and_non_empty_in_the_schema(self) -> None:
        """A claim without support must not be expressible in the model's output."""
        provider, seen = make(tool_response([]))
        provider.synthesize(QUESTION, EVIDENCE)
        item = seen[0]["tools"][0]["input_schema"]["properties"]["claims"]["items"]
        assert item["required"] == ["claim", "evidence_ids"]
        assert item["properties"]["evidence_ids"]["minItems"] == 1

    def test_the_prompt_carries_real_chunk_ids_not_positions(self) -> None:
        provider, seen = make(tool_response([]))
        provider.synthesize(QUESTION, EVIDENCE)
        prompt = seen[0]["messages"][0]["content"]
        assert 'id="pubmed:1:0"' in prompt
        assert 'id="pubmed:2:0"' in prompt
        assert QUESTION in prompt

    def test_no_request_is_made_without_evidence(self) -> None:
        provider, seen = make(tool_response([]))
        assert provider.synthesize(QUESTION, []) == []
        assert seen == []


class TestGrounding:
    def test_returns_claims_that_cite_supplied_evidence(self) -> None:
        provider, _ = make(
            tool_response(
                [
                    {"claim": "Deaminase activity is the cause.", "evidence_ids": ["pubmed:1:0"]},
                    {"claim": "Guide design matters.", "evidence_ids": ["pubmed:2:0"]},
                ]
            )
        )
        citations = provider.synthesize(QUESTION, EVIDENCE)
        assert [c.claim for c in citations] == [
            "Deaminase activity is the cause.",
            "Guide design matters.",
        ]

    def test_a_claim_may_cite_several_passages(self) -> None:
        """The capability extraction lacked, so it is asserted rather than assumed."""
        provider, _ = make(
            tool_response(
                [{"claim": "Both factors matter.", "evidence_ids": ["pubmed:2:0", "pubmed:1:0"]}]
            )
        )
        citations = provider.synthesize(QUESTION, EVIDENCE)
        # Cited in retrieval order, so the strongest passage is named first.
        assert citations[0].evidence_ids == ["pubmed:1:0", "pubmed:2:0"]

    def test_an_invented_id_is_dropped_and_counted(self) -> None:
        provider, _ = make(
            tool_response(
                [{"claim": "Partly real.", "evidence_ids": ["pubmed:1:0", "pubmed:9999:0"]}]
            )
        )
        citations = provider.synthesize(QUESTION, EVIDENCE)
        assert citations[0].evidence_ids == ["pubmed:1:0"]
        assert provider.last_usage is not None
        assert provider.last_usage.unsupported_citations == 1

    def test_a_claim_citing_only_invented_ids_is_discarded(self) -> None:
        provider, _ = make(
            tool_response(
                [
                    {"claim": "Fabricated.", "evidence_ids": ["pubmed:9999:0"]},
                    {"claim": "Real.", "evidence_ids": ["pubmed:1:0"]},
                ]
            )
        )
        citations = provider.synthesize(QUESTION, EVIDENCE)
        assert [c.claim for c in citations] == ["Real."]

    def test_wholly_fabricated_output_becomes_an_abstention(self) -> None:
        provider, _ = make(
            tool_response([{"claim": "Fabricated.", "evidence_ids": ["pubmed:9999:0"]}])
        )
        assert provider.synthesize(QUESTION, EVIDENCE) == []

    def test_duplicate_ids_within_a_claim_are_collapsed(self) -> None:
        provider, _ = make(
            tool_response([{"claim": "Once.", "evidence_ids": ["pubmed:1:0", "pubmed:1:0"]}])
        )
        assert provider.synthesize(QUESTION, EVIDENCE)[0].evidence_ids == ["pubmed:1:0"]

    def test_repeated_claims_are_collapsed(self) -> None:
        provider, _ = make(
            tool_response(
                [
                    {"claim": "Same thing.", "evidence_ids": ["pubmed:1:0"]},
                    {"claim": "same thing.", "evidence_ids": ["pubmed:2:0"]},
                ]
            )
        )
        assert len(provider.synthesize(QUESTION, EVIDENCE)) == 1

    def test_blank_and_malformed_entries_are_skipped(self) -> None:
        provider, _ = make(
            tool_response(
                [
                    {"claim": "   ", "evidence_ids": ["pubmed:1:0"]},
                    {"claim": "No ids field."},
                    {"claim": "Ids not a list.", "evidence_ids": "pubmed:1:0"},
                    {"claim": "Good.", "evidence_ids": ["pubmed:1:0"]},
                ]
            )
        )
        assert [c.claim for c in provider.synthesize(QUESTION, EVIDENCE)] == ["Good."]

    def test_an_empty_claim_list_is_a_valid_abstention(self) -> None:
        provider, _ = make(tool_response([]))
        assert provider.synthesize(QUESTION, EVIDENCE) == []

    def test_respects_the_claim_limit(self) -> None:
        claims = [{"claim": f"Claim {i}.", "evidence_ids": ["pubmed:1:0"]} for i in range(10)]
        provider, _ = make(tool_response(claims), max_claims=3)
        assert len(provider.synthesize(QUESTION, EVIDENCE)) == 3


class TestUsage:
    def test_records_tokens_and_latency(self) -> None:
        ticks = iter([10.0, 11.5])
        provider, _ = make(
            tool_response([{"claim": "A.", "evidence_ids": ["pubmed:1:0"]}]),
            timer=lambda: next(ticks),
        )
        provider.synthesize(QUESTION, EVIDENCE)
        usage = provider.last_usage
        assert usage is not None
        assert (usage.input_tokens, usage.output_tokens) == (100, 50)
        assert usage.latency_seconds == pytest.approx(1.5)

    def test_usage_is_none_before_any_call(self) -> None:
        provider, _ = make(tool_response([]))
        assert provider.last_usage is None


class TestFailure:
    def test_a_provider_error_becomes_a_synthesis_error(self) -> None:
        provider, _ = make(status=400)
        with pytest.raises(SynthesisError, match="language model provider failed"):
            provider.synthesize(QUESTION, EVIDENCE)

    def test_a_response_without_a_tool_call_is_rejected(self) -> None:
        """``tool_choice`` forces the call, so prose means a broken contract."""
        body = tool_response([])
        body["content"] = [{"type": "text", "text": "Here is a paragraph instead."}]
        provider, _ = make(body)
        with pytest.raises(SynthesisError, match="required format"):
            provider.synthesize(QUESTION, EVIDENCE)

    def test_a_reencoded_claims_payload_is_unwrapped(self) -> None:
        """Regression: a live model returned the array as a doubly-wrapped string.

        The real payload was {"claims": "{\\"claims\\": []}"} on an abstention.
        Rejecting it turned a correct abstention into an error.
        """
        body = tool_response([])
        body["content"] = [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": TOOL_NAME,
                "input": {"claims": '{"claims": []}'},
            }
        ]
        provider, _ = make(body)
        assert provider.synthesize(QUESTION, EVIDENCE) == []

    def test_a_singly_encoded_claims_string_is_unwrapped(self) -> None:
        body = tool_response([])
        body["content"] = [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": TOOL_NAME,
                "input": {"claims": '[{"claim": "Real.", "evidence_ids": ["pubmed:1:0"]}]'},
            }
        ]
        provider, _ = make(body)
        assert [c.claim for c in provider.synthesize(QUESTION, EVIDENCE)] == ["Real."]

    def test_a_claims_string_that_is_not_json_is_still_rejected(self) -> None:
        body = tool_response([])
        body["content"] = [
            {"type": "tool_use", "id": "toolu_01", "name": TOOL_NAME, "input": {"claims": "nope"}}
        ]
        provider, _ = make(body)
        with pytest.raises(SynthesisError, match="required format"):
            provider.synthesize(QUESTION, EVIDENCE)

    def test_a_tool_call_without_a_claims_array_is_rejected(self) -> None:
        body = tool_response([])
        body["content"] = [
            {"type": "tool_use", "id": "toolu_01", "name": TOOL_NAME, "input": {"other": 1}}
        ]
        provider, _ = make(body)
        with pytest.raises(SynthesisError, match="required format"):
            provider.synthesize(QUESTION, EVIDENCE)

    def test_a_missing_sdk_is_reported_with_an_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fail(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic":
                raise ImportError("no module named anthropic")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail)
        with pytest.raises(SynthesisError, match="uv sync --extra llm"):
            LlmSynthesizer(api_key="k")

    def test_the_caught_error_type_is_the_sdk_base_class(self) -> None:
        """Guards against an SDK rename silently letting provider errors escape."""
        assert issubclass(anthropic.APIError, anthropic.AnthropicError)
