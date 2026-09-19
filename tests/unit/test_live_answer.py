"""Tests for the live PubMed pipeline.

Nothing is stored, so the properties worth asserting are about what the answer
carries with it: the papers behind its citations, an honest account of what was
searched, and a refusal that is distinguishable from an outage.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.application.live_answer import (
    ABSTAIN_NO_CLAIM,
    ABSTAIN_NO_PAPERS,
    FRESHNESS_LIMITATION,
    answer_live,
)
from bio_research_copilot.domain import Citation, Paper, RetrievedEvidence, SearchQuery

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
QUESTION = "What is the double nicking strategy for reducing off-target cleavage?"


class StubSource:
    def __init__(self, papers: list[Paper] | None = None, error: Exception | None = None) -> None:
        self.papers = RECORDED if papers is None else papers
        self.error = error
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery, term: str | None = None) -> list[Paper]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.papers[: query.max_results]


class SilentSynthesizer:
    name = "silent-v1"

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        return []


class TestAnsweringLive:
    def test_answers_from_papers_the_source_returned(self) -> None:
        result = answer_live(StubSource(), QUESTION)
        assert result.answer.citations
        assert result.papers_fetched == len(RECORDED)
        assert result.passages_indexed > result.papers_fetched

    def test_carries_the_papers_behind_its_citations(self) -> None:
        """Nothing is stored, so an answer that lost its papers is unreadable."""
        result = answer_live(StubSource(), QUESTION)
        cited = {
            item.passage.source_id
            for item in result.answer.evidence
            if any(item.passage.chunk_id in c.evidence_ids for c in result.answer.citations)
        }
        assert cited
        assert set(result.papers) == cited
        for paper in result.papers.values():
            assert paper.title
            assert paper.url.startswith("http")

    def test_claims_are_grounded_in_carried_evidence(self) -> None:
        result = answer_live(StubSource(), QUESTION)
        available = {item.passage.chunk_id for item in result.answer.evidence}
        for citation in result.answer.citations:
            assert set(citation.evidence_ids) <= available

    def test_asks_the_source_for_the_requested_number_of_papers(self) -> None:
        source = StubSource()
        answer_live(source, QUESTION, papers=5)
        assert source.queries[0].max_results == 5

    def test_records_the_methods_used(self) -> None:
        result = answer_live(StubSource(), QUESTION)
        assert result.answer.retrieval_method == "bm25-v1"
        assert result.answer.synthesis_method == "extractive-v1"

    def test_says_the_evidence_was_fetched_just_now(self) -> None:
        """Freshness is a real caveat for a pipeline with no stored index."""
        result = answer_live(StubSource(), QUESTION)
        assert FRESHNESS_LIMITATION in result.answer.limitations

    def test_reports_how_many_papers_were_ranked(self) -> None:
        result = answer_live(StubSource(), QUESTION)
        assert any("PubMed returned" in note for note in result.answer.limitations)


class TestAbstention:
    def test_abstains_when_the_source_finds_nothing(self) -> None:
        result = answer_live(StubSource(papers=[]), QUESTION)
        assert result.answer.abstained
        assert ABSTAIN_NO_PAPERS in result.answer.limitations
        assert result.papers == {}

    def test_abstains_when_synthesis_makes_no_claim(self) -> None:
        result = answer_live(StubSource(), QUESTION, synthesizer=SilentSynthesizer())
        assert result.answer.abstained
        assert ABSTAIN_NO_CLAIM in result.answer.limitations

    def test_an_abstention_still_reports_what_was_searched(self) -> None:
        result = answer_live(StubSource(), QUESTION, synthesizer=SilentSynthesizer())
        assert result.papers_fetched == len(RECORDED)


class TestFailures:
    @pytest.mark.parametrize(
        "error",
        [LiteratureSourceUnavailableError("down"), LiteratureSourceParseError("bad xml")],
    )
    def test_a_source_failure_propagates_rather_than_abstaining(self, error: Exception) -> None:
        """An outage presented as a refusal would be a lie to the reader."""
        with pytest.raises(type(error)):
            answer_live(StubSource(error=error), QUESTION)

    def test_a_blank_question_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            answer_live(StubSource(), "   ")
