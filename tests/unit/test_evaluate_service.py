"""Tests for the evaluation use case on a tiny store with hand-set judgments."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import evaluate_retrieval, load_corpus
from bio_research_copilot.application.evaluate import collapse_to_papers
from bio_research_copilot.domain import Passage, RetrievedEvidence, Source
from bio_research_copilot.evaluation import Benchmark

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)


def bench(questions: list[dict[str, object]]) -> Benchmark:
    return Benchmark.model_validate(
        {"version": "test", "judgment_unit": "paper", "questions": questions}
    )


@pytest.fixture
def repo() -> SqliteRepository:
    repo = SqliteRepository.open(":memory:")
    load_corpus(repo, RECORDED, label="fixture", clock=lambda: T1)
    return repo


def evidence(chunk: str, rank: int) -> RetrievedEvidence:
    pid, pos = chunk.split(":")
    return RetrievedEvidence(
        passage=Passage(
            source=Source.PUBMED,
            source_id=pid,
            chunk_id=f"pubmed:{chunk}",
            position=int(pos),
            section="abstract",
            text="x",
            content_hash="0" * 64,
            strategy="t",
        ),
        rank=rank,
        score=1.0 / rank,
        method="t",
    )


def test_collapse_keeps_best_rank_per_paper_and_caps_at_k() -> None:
    ranked = [evidence("1:0", 1), evidence("2:0", 2), evidence("1:1", 3), evidence("3:0", 4)]
    assert collapse_to_papers(ranked, k=2) == ["1", "2"]
    assert collapse_to_papers(ranked, k=10) == ["1", "2", "3"]
    assert collapse_to_papers([], k=3) == []


def test_scores_a_judged_question(repo: SqliteRepository) -> None:
    b = bench(
        [
            {
                "id": "a",
                "topic": "crispr",
                "type": "mechanism",
                "question": "double nicking strategy off-target cleavage",
                "relevant": ["24157548"],
            }
        ]
    )
    report = evaluate_retrieval(repo, b, k=3, clock=lambda: T1)
    assert report.retriever == "bm25-v1"
    assert report.corpus_papers == 3
    assert report.corpus_passages == 6
    assert report.judged_questions == 1
    assert report.unjudged_questions == []
    result = report.results[0]
    assert result.ranked_papers[0] == "24157548"
    assert result.metrics is not None
    assert result.metrics.reciprocal_rank == 1.0
    assert result.metrics.recall == 1.0
    assert result.metrics.precision == pytest.approx(1 / 3)
    assert report.mean_mrr == 1.0
    assert report.by_type["mechanism"].questions == 1


def test_unjudged_questions_are_listed_and_excluded_from_means(repo: SqliteRepository) -> None:
    b = bench(
        [
            {"id": "a", "topic": "t", "type": "target", "question": "CRISPR", "relevant": []},
            {
                "id": "b",
                "topic": "t",
                "type": "target",
                "question": "Cas9",
                "relevant": ["28375731"],
            },
        ]
    )
    report = evaluate_retrieval(repo, b, k=3, clock=lambda: T1)
    assert report.unjudged_questions == ["a"]
    assert report.judged_questions == 1
    assert report.results[0].metrics is None
    assert report.results[0].judged is False


def test_unanswerable_questions_report_what_came_back(repo: SqliteRepository) -> None:
    b = bench(
        [
            {"id": "u1", "topic": "none", "type": "unanswerable", "question": "zebrafish"},
            {"id": "u2", "topic": "none", "type": "unanswerable", "question": "CRISPR genome"},
        ]
    )
    report = evaluate_retrieval(repo, b, k=3, clock=lambda: T1)
    assert report.judged_questions == 0
    assert report.mean_mrr is None
    by_id = {u.id: u for u in report.unanswerable}
    assert by_id["u1"].papers_returned == 0
    assert by_id["u1"].top_score is None
    assert by_id["u2"].papers_returned > 0
    assert by_id["u2"].top_score is not None


def test_latency_uses_injected_timer(repo: SqliteRepository) -> None:
    ticks = iter([0.0, 0.002, 1.0, 1.001])
    b = bench(
        [
            {
                "id": "a",
                "topic": "t",
                "type": "target",
                "question": "Cas9",
                "relevant": ["28375731"],
            },
            {
                "id": "b",
                "topic": "t",
                "type": "target",
                "question": "Cas9",
                "relevant": ["28375731"],
            },
        ]
    )
    report = evaluate_retrieval(repo, b, k=3, clock=lambda: T1, timer=lambda: next(ticks))
    assert [r.latency_ms for r in report.results] == [pytest.approx(2.0), pytest.approx(1.0)]
    assert report.mean_latency_ms == pytest.approx(1.5)


def test_report_serialises_to_json(repo: SqliteRepository) -> None:
    b = bench([{"id": "u1", "topic": "none", "type": "unanswerable", "question": "zebrafish"}])
    report = evaluate_retrieval(repo, b, k=3, clock=lambda: T1)
    dumped = report.model_dump(mode="json")
    assert dumped["generated_at"].startswith("2026-09-17")
    assert dumped["results"][0]["metrics"] is None
