"""Tests for the synthesis evaluation harness.

The harness reports numbers a reader will trust without re-deriving them, so
the tests are mostly about what the numbers mean rather than that arithmetic
works: that a correct refusal is not scored as a wrong citation, that the two
kinds of abstention error stay separate, and that the dumped claims carry enough
context to be judged without the database.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import load_corpus
from bio_research_copilot.application.evaluate_synthesis import evaluate_synthesis
from bio_research_copilot.domain import Citation, RetrievedEvidence
from bio_research_copilot.evaluation import Benchmark
from bio_research_copilot.synthesis.llm import Usage

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)

ANSWERABLE = "What is the double nicking strategy for reducing off-target cleavage?"
UNANSWERABLE = "zebrafish husbandry temperature"


@pytest.fixture
def repo() -> SqliteRepository:
    repo = SqliteRepository.open(":memory:")
    load_corpus(repo, RECORDED, label="fixture", clock=lambda: T1)
    return repo


@pytest.fixture
def relevant_ids() -> list[str]:
    return [paper.source_id for paper in RECORDED[:2]]


def benchmark(questions: list[dict[str, object]]) -> Benchmark:
    return Benchmark.model_validate(
        {"version": "test-v1", "judgment_unit": "paper", "questions": questions}
    )


def answerable(qid: str, relevant: list[str]) -> dict[str, object]:
    return {
        "id": qid,
        "topic": "crispr",
        "type": "mechanism",
        "question": ANSWERABLE,
        "relevant": relevant,
    }


def unanswerable(qid: str) -> dict[str, object]:
    return {"id": qid, "topic": "none", "type": "unanswerable", "question": UNANSWERABLE}


class CountingSynthesizer:
    """A provider that reports usage, so cost accounting can be asserted."""

    name = "fake-llm-v1"

    def __init__(self, *, unsupported: int = 0, claims: int = 1) -> None:
        self.unsupported = unsupported
        self.claims = claims
        self.calls = 0
        self.last_usage: Usage | None = None

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        self.calls += 1
        self.last_usage = Usage(
            input_tokens=100,
            output_tokens=20,
            latency_seconds=0.1,
            unsupported_citations=self.unsupported,
        )
        return [
            Citation(claim=f"Claim {i}.", evidence_ids=[evidence[0].passage.chunk_id])
            for i in range(self.claims)
        ]


class TestAbstentionAccounting:
    def test_separates_the_two_kinds_of_error(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(
            repo, benchmark([answerable("q1", relevant_ids), unanswerable("q2")])
        )
        a = report.abstention
        assert (a.answerable, a.answered, a.false_abstentions) == (1, 1, 0)
        assert (a.unanswerable, a.abstained, a.false_answers) == (1, 1, 0)

    def test_counts_a_refused_answerable_question_as_a_false_abstention(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(
            repo, benchmark([answerable("q1", relevant_ids)]), min_score=1e9
        )
        assert report.abstention.false_abstentions == 1
        assert report.abstention.answered == 0

    def test_records_why_each_abstention_happened(self, repo: SqliteRepository) -> None:
        report = evaluate_synthesis(repo, benchmark([unanswerable("q1")]))
        assert report.results[0].abstain_reason


class TestCitationCorrectness:
    def test_scores_cited_papers_against_the_judgments(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(repo, benchmark([answerable("q1", relevant_ids)]))
        result = report.results[0]
        assert result.citation_precision is not None
        expected = len([p for p in result.cited_papers if p in relevant_ids]) / len(
            result.cited_papers
        )
        assert result.citation_precision == pytest.approx(expected)

    def test_citing_nothing_relevant_scores_zero_not_none(self, repo: SqliteRepository) -> None:
        report = evaluate_synthesis(repo, benchmark([answerable("q1", ["99999999"])]))
        assert report.results[0].citation_precision == 0.0

    def test_an_abstention_is_not_scored_as_a_bad_citation(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        """A correct refusal must not drag the citation average down."""
        report = evaluate_synthesis(
            repo, benchmark([answerable("q1", relevant_ids), unanswerable("q2")])
        )
        assert report.results[1].citation_precision is None
        assert report.mean_citation_precision == report.results[0].citation_precision

    def test_mean_is_none_when_nothing_was_ever_cited(self, repo: SqliteRepository) -> None:
        report = evaluate_synthesis(repo, benchmark([unanswerable("q1")]))
        assert report.mean_citation_precision is None


class TestCostAndFabrication:
    def test_totals_tokens_across_questions(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        provider = CountingSynthesizer()
        report = evaluate_synthesis(
            repo,
            benchmark([answerable("q1", relevant_ids), answerable("q2", relevant_ids)]),
            synthesizer=provider,
        )
        assert report.total_input_tokens == 200
        assert report.total_output_tokens == 40

    def test_counts_fabrication_attempts(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        provider = CountingSynthesizer(unsupported=2)
        report = evaluate_synthesis(
            repo, benchmark([answerable("q1", relevant_ids)]), synthesizer=provider
        )
        assert report.unsupported_citations == 2

    def test_a_provider_without_usage_reports_zero_cost(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        """The extractive baseline has no usage record and must not break the report."""
        report = evaluate_synthesis(repo, benchmark([answerable("q1", relevant_ids)]))
        assert report.total_input_tokens == 0
        assert report.unsupported_citations == 0

    def test_calls_the_provider_once_per_question(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        """Guards the report against paying twice for the same numbers."""
        provider = CountingSynthesizer()
        evaluate_synthesis(
            repo,
            benchmark([answerable("q1", relevant_ids), answerable("q2", relevant_ids)]),
            synthesizer=provider,
        )
        assert provider.calls == 2


class TestDumpedClaims:
    def test_each_claim_carries_the_text_it_cites(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        """Faithfulness judging must not need the database that produced the answer."""
        report = evaluate_synthesis(repo, benchmark([answerable("q1", relevant_ids)]))
        claims = report.results[0].claims
        assert claims
        for record in claims:
            assert record.cited
            for passage in record.cited:
                assert passage.text
                assert passage.chunk_id in record.evidence_ids

    def test_records_the_methods_that_produced_the_answers(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(repo, benchmark([answerable("q1", relevant_ids)]))
        assert report.retriever == "bm25-v1"
        assert report.synthesizer == "extractive-v1"

    def test_report_serialises_to_json(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(repo, benchmark([answerable("q1", relevant_ids)]))
        dumped = report.model_dump(mode="json")
        assert dumped["results"][0]["claims"][0]["cited"][0]["text"]


class TestByType:
    def test_groups_results_by_question_type(
        self, repo: SqliteRepository, relevant_ids: list[str]
    ) -> None:
        report = evaluate_synthesis(
            repo, benchmark([answerable("q1", relevant_ids), unanswerable("q2")])
        )
        assert set(report.by_type) == {"mechanism", "unanswerable"}
        assert report.by_type["unanswerable"].answered == 0
