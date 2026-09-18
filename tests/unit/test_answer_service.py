"""Tests for the end-to-end answer use case, including the grounding guarantee."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import answer_question, load_corpus, paper_ids_cited
from bio_research_copilot.application.answer import (
    ABSTAIN_BELOW_FLOOR,
    ABSTAIN_NO_CLAIM,
    ABSTAIN_NO_RESULTS,
)
from bio_research_copilot.domain import Citation, ResearchAnswer, RetrievedEvidence

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
QUESTION = "What is the double nicking strategy for reducing off-target cleavage?"


@pytest.fixture
def repo() -> SqliteRepository:
    repo = SqliteRepository.open(":memory:")
    load_corpus(repo, RECORDED, label="fixture", clock=lambda: T1)
    return repo


class FakeSynthesizer:
    """Returns claims the caller dictates, including invalid ones."""

    name = "fake-synth-v1"

    def __init__(self, citations: list[Citation] | None = None) -> None:
        self.citations = citations
        self.seen: list[list[RetrievedEvidence]] = []

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        self.seen.append(list(evidence))
        if self.citations is not None:
            return self.citations
        return [Citation(claim="A claim.", evidence_ids=[evidence[0].passage.chunk_id])]


class TestGrounding:
    def test_every_claim_cites_evidence_carried_by_the_answer(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION)
        assert answer.citations
        available = {item.passage.chunk_id for item in answer.evidence}
        for citation in answer.citations:
            assert set(citation.evidence_ids) <= available

    def test_claims_are_verbatim_from_the_cited_passage(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION)
        texts = {item.passage.chunk_id: item.passage.text for item in answer.evidence}
        for citation in answer.citations:
            assert citation.claim in texts[citation.evidence_ids[0]]

    def test_an_answer_citing_absent_evidence_cannot_be_constructed(self) -> None:
        """The Stage 5 gate, enforced by the model rather than by a caller's check."""
        with pytest.raises(ValidationError, match="not in this answer"):
            ResearchAnswer(
                question="q",
                citations=[Citation(claim="c", evidence_ids=["pubmed:does-not-exist:0"])],
                evidence=[],
                retrieval_method="t",
                synthesis_method="t",
            )

    def test_a_fabricating_provider_is_caught(self, repo: SqliteRepository) -> None:
        """Whatever produced the claim, an uncitable one cannot reach the caller."""
        rogue = FakeSynthesizer([Citation(claim="Invented.", evidence_ids=["pubmed:9999:0"])])
        with pytest.raises(ValidationError, match="not in this answer"):
            answer_question(repo, QUESTION, synthesizer=rogue)

    def test_an_abstained_answer_cannot_carry_citations(self) -> None:
        with pytest.raises(ValidationError, match="abstained answer cannot carry citations"):
            ResearchAnswer(
                question="q",
                citations=[Citation(claim="c", evidence_ids=["x"])],
                evidence=[],
                abstained=True,
                retrieval_method="t",
                synthesis_method="t",
            )


class TestAbstention:
    def test_abstains_when_retrieval_finds_nothing(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, "zebrafish husbandry temperature")
        assert answer.abstained
        assert answer.citations == []
        assert ABSTAIN_NO_RESULTS in answer.limitations

    def test_abstains_below_the_score_floor(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION, min_score=1e9)
        assert answer.abstained
        assert ABSTAIN_BELOW_FLOOR in answer.limitations
        # The evidence is still reported, so a caller can see what was rejected.
        assert answer.evidence

    def test_does_not_abstain_above_the_floor(self, repo: SqliteRepository) -> None:
        assert not answer_question(repo, QUESTION, min_score=0.0).abstained

    def test_abstains_when_synthesis_makes_no_claim(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION, synthesizer=FakeSynthesizer([]))
        assert answer.abstained
        assert ABSTAIN_NO_CLAIM in answer.limitations

    def test_an_abstention_still_explains_itself(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, "zebrafish husbandry temperature")
        assert len(answer.limitations) >= 2


class TestEvidenceAndProvenance:
    def test_only_budgeted_evidence_reaches_synthesis_and_the_answer(
        self, repo: SqliteRepository
    ) -> None:
        synth = FakeSynthesizer()
        answer = answer_question(repo, QUESTION, synthesizer=synth, char_budget=200)
        assert synth.seen[0] == answer.evidence
        assert len(answer.evidence) < 6

    def test_records_both_methods(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION)
        assert answer.retrieval_method == "bm25-v1"
        assert answer.synthesis_method == "extractive-v1"

    def test_limitations_report_the_evidence_base(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION)
        assert any("passages from" in note for note in answer.limitations)
        assert any("titles and abstracts" in note for note in answer.limitations)

    def test_single_paper_evidence_is_flagged_as_uncorroborated(
        self, repo: SqliteRepository
    ) -> None:
        answer = answer_question(repo, QUESTION, evidence_limit=1)
        assert any("single paper" in note for note in answer.limitations)

    def test_cited_paper_ids_follow_citation_order(self, repo: SqliteRepository) -> None:
        answer = answer_question(repo, QUESTION)
        ids = paper_ids_cited(answer)
        assert ids
        assert len(ids) == len(set(ids))
        first_chunk = answer.citations[0].evidence_ids[0]
        assert ids[0] == first_chunk.split(":")[1]

    def test_blank_question_is_rejected(self, repo: SqliteRepository) -> None:
        with pytest.raises(ValidationError):
            answer_question(repo, "   ")

    def test_answer_serialises_to_json(self, repo: SqliteRepository) -> None:
        dumped = answer_question(repo, QUESTION).model_dump(mode="json")
        assert dumped["abstained"] is False
        assert dumped["citations"][0]["evidence_ids"]
        assert dumped["evidence"][0]["passage"]["chunk_id"]
