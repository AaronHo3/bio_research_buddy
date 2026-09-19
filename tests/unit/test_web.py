"""Tests for the demo HTTP surface.

The properties that matter here are the ones protecting a public URL from an
unbounded bill, so they are asserted rather than trusted: spending stops at the
budget, a repeated question is not paid for twice, one address cannot hammer the
endpoint, and an exhausted budget degrades to the free provider instead of
failing. The literature source is always a stub, so no test reaches NCBI.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bio_research_copilot.adapters.literature import LiteratureSourceUnavailableError
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import load_corpus
from bio_research_copilot.application.demo_export import export_demo
from bio_research_copilot.application.evaluate_synthesis import evaluate_synthesis
from bio_research_copilot.domain import Citation, Paper, RetrievedEvidence, SearchQuery
from bio_research_copilot.evaluation import Benchmark
from bio_research_copilot.synthesis import SynthesisError
from bio_research_copilot.synthesis.llm import Usage
from bio_research_copilot.web.app import create_app, load_demo_data

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = parse_efetch_xml((FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8"))
T1 = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
ANSWERABLE = "What is the double nicking strategy for reducing off-target cleavage?"


class StubSource:
    """Stands in for PubMed. No test in this module touches the network."""

    def __init__(self, papers: list[Paper] | None = None, error: Exception | None = None) -> None:
        self.papers = RECORDED if papers is None else papers
        self.error = error
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery, term: str | None = None) -> list[Paper]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.papers[: query.max_results]


class StubSynthesizer:
    """A model-backed provider that reports a fixed cost."""

    name = "anthropic:stub-model"

    def __init__(self, *, tokens: int = 1000, fail: bool = False) -> None:
        self.tokens = tokens
        self.fail = fail
        self.calls = 0
        self.last_usage: Usage | None = None

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        self.calls += 1
        if self.fail:
            raise SynthesisError("provider exploded")
        self.last_usage = Usage(input_tokens=self.tokens, output_tokens=0, latency_seconds=0.1)
        return [
            Citation(claim="A model-written claim.", evidence_ids=[evidence[0].passage.chunk_id])
        ]


@pytest.fixture
def data_path(tmp_path: Path) -> Path:
    benchmark = Benchmark.model_validate(
        {
            "version": "t",
            "judgment_unit": "paper",
            "questions": [
                {
                    "id": "q1",
                    "topic": "crispr",
                    "type": "mechanism",
                    "question": ANSWERABLE,
                    "relevant": [RECORDED[0].source_id],
                },
                {
                    "id": "q2",
                    "topic": "none",
                    "type": "unanswerable",
                    "question": "zebrafish husbandry temperature",
                },
            ],
        }
    )
    with SqliteRepository.open(tmp_path / "c.sqlite") as repo:
        load_corpus(repo, RECORDED, label="fixture", clock=lambda: T1)
        report = evaluate_synthesis(repo, benchmark)
        data = export_demo(repo, report, clock=lambda: T1)
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(data.model_dump(mode="json")), encoding="utf-8")
    return path


def build(
    data_path: Path, *, source: StubSource | None = None, **kwargs: Any
) -> tuple[TestClient, StubSource]:
    stub = source or StubSource()
    app = create_app(source=stub, data_path=data_path, **kwargs)
    return TestClient(app), stub


@pytest.fixture
def client(data_path: Path) -> TestClient:
    return build(data_path)[0]


class TestExamples:
    def test_serves_the_recorded_answers(self, client: TestClient) -> None:
        body = client.get("/api/examples").json()
        assert len(body["answers"]) == 2

    def test_every_cited_paper_is_listed_as_a_source(self, client: TestClient) -> None:
        for answer in client.get("/api/examples").json()["answers"]:
            listed = {s["paper_id"] for s in answer["sources"]}
            for claim in answer["claims"]:
                assert set(claim["paper_ids"]) <= listed

    def test_carries_the_disclaimer(self, client: TestClient) -> None:
        assert "Not medical advice" in client.get("/api/examples").json()["disclaimer"]


class TestLiveAnswer:
    def test_answers_from_the_literature_source(self, data_path: Path) -> None:
        client, source = build(data_path)
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["claims"]
        assert body["sources"]
        assert source.queries, "the source was never searched"

    def test_reports_how_many_papers_were_fetched(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["papers_fetched"] == len(RECORDED)

    def test_claims_cite_only_listed_sources(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        listed = {s["paper_id"] for s in body["sources"]}
        for claim in body["claims"]:
            assert set(claim["paper_ids"]) <= listed

    def test_abstains_visibly(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": "zebrafish husbandry"}).json()
        assert body["abstained"] is True
        assert body["claims"] == []
        assert body["abstain_reason"]

    @pytest.mark.parametrize("question", ["", "   ", "x" * 5000])
    def test_bad_questions_are_rejected(self, client: TestClient, question: str) -> None:
        assert client.post("/api/answer", json={"question": question}).status_code == 422

    def test_an_unreachable_source_is_a_503_not_an_abstention(self, data_path: Path) -> None:
        """An outage must not be dressed up as a considered refusal."""
        source = StubSource(error=LiteratureSourceUnavailableError("NCBI down"))
        client, _ = build(data_path, source=source)
        response = client.post("/api/answer", json={"question": ANSWERABLE})
        assert response.status_code == 503
        assert "PubMed" in response.json()["detail"]

    def test_a_failing_provider_is_a_502(self, data_path: Path) -> None:
        client, _ = build(
            data_path, build_synthesizer=lambda: StubSynthesizer(fail=True), daily_tokens=10_000
        )
        assert client.post("/api/answer", json={"question": ANSWERABLE}).status_code == 502


class TestCitedPassages:
    """The page lets a reader open the exact passage behind a claim.

    That is the product's whole promise made checkable, so the text has to
    travel with the answer rather than being fetched again or trusted.
    """

    def test_every_cited_chunk_has_its_text(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["claims"]
        for claim in body["claims"]:
            for chunk_id in claim["chunk_ids"]:
                assert body["passages"].get(chunk_id)

    def test_uncited_evidence_is_not_shipped(self, client: TestClient) -> None:
        """Only what a claim rests on, so the payload stays small and relevant."""
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        cited = {c for claim in body["claims"] for c in claim["chunk_ids"]}
        assert set(body["passages"]) == cited

    def test_an_abstention_ships_no_passages(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": "zebrafish husbandry"}).json()
        assert body["passages"] == {}

    def test_recorded_examples_carry_them_too(self, client: TestClient) -> None:
        for answer in client.get("/api/examples").json()["answers"]:
            for claim in answer["claims"]:
                for chunk_id in claim["chunk_ids"]:
                    assert answer["passages"].get(chunk_id)


class TestSecrets:
    """A public URL must not hand a visitor anything about the provider.

    The API key is the obvious thing to check, but the provider's own error text
    is the likelier leak: it names internal plumbing and a visitor cannot act on
    it, so it belongs in a log rather than a response body.
    """

    def test_a_provider_failure_says_nothing_about_the_provider(self, data_path: Path) -> None:
        client, _ = build(
            data_path, build_synthesizer=lambda: StubSynthesizer(fail=True), daily_tokens=10_000
        )
        response = client.post("/api/answer", json={"question": ANSWERABLE})
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert "provider exploded" not in detail
        assert detail == "The answer could not be written just now. Please try again."

    def test_health_exposes_no_credentials(self, data_path: Path) -> None:
        client, _ = build(
            data_path, build_synthesizer=lambda: StubSynthesizer(), daily_tokens=10_000
        )
        body = client.get("/api/health").text
        assert "api_key" not in body.lower()
        assert "sk-ant" not in body


class TestBudget:
    def test_uses_the_model_while_budget_remains(self, data_path: Path) -> None:
        client, _ = build(
            data_path, build_synthesizer=lambda: StubSynthesizer(), daily_tokens=10_000
        )
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["model_backed"] is True
        assert body["synthesizer"] == "anthropic:stub-model"

    def test_degrades_to_extractive_once_spent(self, data_path: Path) -> None:
        """The whole point: running out costs a worse answer, not an outage."""
        client, _ = build(
            data_path,
            build_synthesizer=lambda: StubSynthesizer(tokens=1200),
            daily_tokens=1000,
        )
        first = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert first["model_backed"] is True

        second = client.post("/api/answer", json={"question": "base editing off target"}).json()
        assert second["model_backed"] is False
        assert second["synthesizer"] == "extractive-v1"
        assert any("budget" in note.lower() for note in second["limitations"])

    def test_an_abstention_is_not_reported_as_model_backed(self, data_path: Path) -> None:
        """Retrieval can abstain before synthesis runs; no model wrote that."""
        provider = StubSynthesizer()
        client, _ = build(data_path, build_synthesizer=lambda: provider, daily_tokens=10_000)
        body = client.post("/api/answer", json={"question": "zebrafish husbandry"}).json()
        assert body["abstained"] is True
        assert body["model_backed"] is False
        assert provider.calls == 0

    def test_the_ceiling_can_be_overshot_by_at_most_one_call(self, data_path: Path) -> None:
        """Cost is charged from real usage afterwards, never estimated beforehand.

        A call that starts under the limit is allowed to finish, so a day can end
        slightly over. Pinned here because it is a deliberate trade, not a bug.
        """
        client, _ = build(
            data_path,
            build_synthesizer=lambda: StubSynthesizer(tokens=900),
            daily_tokens=1000,
        )
        assert client.post("/api/answer", json={"question": "base editor off target"}).json()[
            "model_backed"
        ]
        # 900 spent, still under 1000, so this one is permitted and takes it to 1800.
        assert client.post("/api/answer", json={"question": "cas9 nickase specificity"}).json()[
            "model_backed"
        ]
        assert not client.post("/api/answer", json={"question": "guide rna design editing"}).json()[
            "model_backed"
        ]
        assert client.get("/api/health").json()["budget"]["spent"] == 1800

    def test_zero_budget_never_calls_the_model(self, data_path: Path) -> None:
        provider = StubSynthesizer()
        client, _ = build(data_path, build_synthesizer=lambda: provider, daily_tokens=0)
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["model_backed"] is False
        assert provider.calls == 0

    def test_no_key_configured_answers_extractively(self, client: TestClient) -> None:
        body = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert body["model_backed"] is False
        assert body["synthesizer"] == "extractive-v1"

    def test_health_reports_remaining_budget(self, data_path: Path) -> None:
        client, _ = build(
            data_path, build_synthesizer=lambda: StubSynthesizer(tokens=400), daily_tokens=1000
        )
        client.post("/api/answer", json={"question": ANSWERABLE})
        budget = client.get("/api/health").json()["budget"]
        assert budget["spent"] == 400
        assert budget["remaining"] == 600


class TestCache:
    def test_a_repeated_question_is_not_paid_for_twice(self, data_path: Path) -> None:
        provider = StubSynthesizer()
        client, _ = build(data_path, build_synthesizer=lambda: provider, daily_tokens=10_000)
        client.post("/api/answer", json={"question": ANSWERABLE})
        second = client.post("/api/answer", json={"question": ANSWERABLE}).json()
        assert provider.calls == 1
        assert second["cached"] is True

    def test_wording_differences_that_do_not_matter_still_hit(self, data_path: Path) -> None:
        client, source = build(data_path)
        client.post("/api/answer", json={"question": ANSWERABLE})
        client.post("/api/answer", json={"question": f"  {ANSWERABLE.upper()}  "})
        assert len(source.queries) == 1

    def test_a_different_question_is_a_miss(self, data_path: Path) -> None:
        client, source = build(data_path)
        client.post("/api/answer", json={"question": ANSWERABLE})
        client.post("/api/answer", json={"question": "something else entirely"})
        assert len(source.queries) == 2


class TestRateLimit:
    def test_one_client_is_cut_off_after_the_limit(self, data_path: Path) -> None:
        client, _ = build(data_path, rate_limit=2, rate_window=3600.0)
        assert client.post("/api/answer", json={"question": "alpha editing"}).status_code == 200
        assert client.post("/api/answer", json={"question": "beta editing"}).status_code == 200
        third = client.post("/api/answer", json={"question": "gamma editing"})
        assert third.status_code == 429
        assert "Too many" in third.json()["detail"]

    def test_the_limit_is_counted_before_any_work(self, data_path: Path) -> None:
        client, source = build(data_path, rate_limit=1, rate_window=3600.0)
        client.post("/api/answer", json={"question": "alpha editing"})
        client.post("/api/answer", json={"question": "beta editing"})
        assert len(source.queries) == 1


class TestPageAndHealth:
    def test_health_reports_model_availability(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["model_available"] is False

    def test_serves_the_page(self, client: TestClient) -> None:
        """Asserts the page works, not how it is worded, so copy can change freely."""
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="ask-form"' in response.text
        assert "/api/answer" in response.text

    def test_missing_export_fails_with_a_command_to_run(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="export-demo"):
            load_demo_data(tmp_path / "absent.json")
