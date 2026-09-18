"""Tests for prompt construction.

Prompt wording is part of the system's behaviour, not a detail, so it is pinned
to a golden file. A change to the instructions then has to be made deliberately
and reviewed as a diff, rather than drifting and showing up later as an
unexplained change in answer quality.
"""

import hashlib
from pathlib import Path

import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, Source
from bio_research_copilot.synthesis.prompt import (
    CLAIM_TOOL,
    SYSTEM_PROMPT,
    TOOL_NAME,
    build_user_prompt,
    format_evidence,
)

GOLDEN = Path(__file__).parent.parent / "fixtures" / "prompt" / "user_prompt.txt"
QUESTION = "What causes off-target editing by base editors?"


def evidence(chunk: str, section: str, text: str, rank: int) -> RetrievedEvidence:
    pmid, position = chunk.split(":")
    return RetrievedEvidence(
        passage=Passage(
            source=Source.PUBMED,
            source_id=pmid,
            chunk_id=f"pubmed:{chunk}",
            position=int(position),
            section=section,
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            strategy="section-v1",
        ),
        rank=rank,
        score=1.0 / rank,
        method="bm25-v1",
    )


@pytest.fixture
def items() -> list[RetrievedEvidence]:
    return [
        evidence("31978945:0", "title", "Base editing: precision chemistry on the genome.", 1),
        evidence(
            "31978945:2",
            "results",
            "Off-target editing arose chiefly from deaminase activity on transiently"
            " exposed single-stranded DNA.",
            2,
        ),
        evidence(
            "29160308:1",
            "background",
            "Guide RNA design influences editing specificity in mammalian cells.",
            3,
        ),
    ]


class TestUserPrompt:
    def test_matches_the_golden_file(self, items: list[RetrievedEvidence]) -> None:
        built = build_user_prompt(QUESTION, items, max_claims=6)
        assert built == GOLDEN.read_text(encoding="utf-8")

    def test_evidence_precedes_the_question(self, items: list[RetrievedEvidence]) -> None:
        """Material first, so the model reads before it knows what to look for."""
        built = build_user_prompt(QUESTION, items, max_claims=6)
        assert built.index("</evidence>") < built.index(QUESTION)

    def test_passages_appear_in_the_order_given(self, items: list[RetrievedEvidence]) -> None:
        built = build_user_prompt(QUESTION, items, max_claims=6)
        positions = [built.index(f'id="{i.passage.chunk_id}"') for i in items]
        assert positions == sorted(positions)

    def test_the_claim_limit_is_stated(self, items: list[RetrievedEvidence]) -> None:
        assert "at most 3 claims" in build_user_prompt(QUESTION, items, max_claims=3)

    def test_no_evidence_still_produces_a_well_formed_prompt(self) -> None:
        built = build_user_prompt(QUESTION, [], max_claims=6)
        assert "<evidence>" in built and "</evidence>" in built


class TestEvidenceBlock:
    def test_labels_each_passage_with_its_real_chunk_id(
        self, items: list[RetrievedEvidence]
    ) -> None:
        """Ids, not positions, so a citation needs no translation step."""
        block = format_evidence(items)
        for item in items:
            assert f'id="{item.passage.chunk_id}"' in block

    def test_carries_the_passage_text_verbatim(self, items: list[RetrievedEvidence]) -> None:
        block = format_evidence(items)
        for item in items:
            assert item.passage.text in block

    def test_empty_evidence_is_an_empty_block(self) -> None:
        assert format_evidence([]) == ""


class TestSystemPrompt:
    @pytest.mark.parametrize(
        "requirement",
        [
            "exact id",
            "Do not add facts from your own knowledge",
            "medical advice",
            "record no claims",
            "standalone sentence",
        ],
    )
    def test_states_the_rules_the_product_depends_on(self, requirement: str) -> None:
        assert requirement in SYSTEM_PROMPT


class TestClaimTool:
    def test_is_named_consistently_with_the_forced_choice(self) -> None:
        assert CLAIM_TOOL["name"] == TOOL_NAME

    def test_a_claim_cannot_be_recorded_without_evidence(self) -> None:
        item = CLAIM_TOOL["input_schema"]["properties"]["claims"]["items"]
        assert set(item["required"]) == {"claim", "evidence_ids"}
        assert item["properties"]["evidence_ids"]["minItems"] == 1

    def test_claims_is_required_so_an_abstention_is_explicit(self) -> None:
        """An empty list is a decision; a missing field would be ambiguous."""
        assert CLAIM_TOOL["input_schema"]["required"] == ["claims"]
