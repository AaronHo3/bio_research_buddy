"""Tests for the PubMed client's HTTP behaviour, fully offline.

``httpx.MockTransport`` lets a plain function stand in for the network. Each
test declares what PubMed "says" and asserts what the client does about it.
A fake clock records every sleep the client asks for, so retry and pacing
behaviour is asserted exactly and the suite never actually waits.
"""

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.adapters.pubmed import PubMedClient
from bio_research_copilot.adapters.resilience import RetryPolicy
from bio_research_copilot.domain import SearchQuery
from tests.unit.fakes import FakeClock

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
ESEARCH = (FIXTURES / "esearch_crispr.json").read_text(encoding="utf-8")
ESEARCH_EMPTY = (FIXTURES / "esearch_empty.json").read_text(encoding="utf-8")
EFETCH = (FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8")

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler,
    clock: FakeClock | None = None,
    retry: RetryPolicy | None = None,
    email: str | None = None,
    api_key: str | None = None,
) -> PubMedClient:
    clock = clock or FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PubMedClient(
        http,
        tool="test-tool",
        email=email,
        api_key=api_key,
        retry=retry,
        clock=clock,
        sleep=clock.sleep,
    )


def canned(
    esearch_body: str = ESEARCH, efetch_body: str = EFETCH
) -> tuple[Handler, list[httpx.Request]]:
    """A handler that serves fixtures and records every request it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, text=esearch_body)
        if request.url.path.endswith("efetch.fcgi"):
            return httpx.Response(200, text=efetch_body)
        return httpx.Response(404)

    return handler, seen


def flaky(failures: list[Exception | int]) -> tuple[Handler, list[httpx.Request]]:
    """Fail with each listed status or exception in turn, then serve fixtures."""
    ok, seen = canned()
    queue = list(failures)

    def handler(request: httpx.Request) -> httpx.Response:
        if queue:
            failure = queue.pop(0)
            seen.append(request)
            if isinstance(failure, int):
                return httpx.Response(failure)
            raise failure
        return ok(request)

    return handler, seen


# -- Happy path ----------------------------------------------------------------


def test_two_step_flow_returns_papers_in_esearch_order() -> None:
    handler, seen = canned()
    query = SearchQuery(raw="CRISPR base editing off-target", max_results=3)
    papers = make_client(handler).search(query)
    assert [r.url.path.rsplit("/", 1)[-1] for r in seen] == ["esearch.fcgi", "efetch.fcgi"]
    assert [p.source_id for p in papers] == ["24157548", "28375731", "25315507"]


def test_efetch_order_is_restored_to_esearch_rank() -> None:
    reversed_ids = ["25315507", "28375731", "24157548"]
    body = json.loads(ESEARCH)
    body["esearchresult"]["idlist"] = reversed_ids
    handler, _ = canned(esearch_body=json.dumps(body))
    papers = make_client(handler).search(SearchQuery(raw="q"))
    assert [p.source_id for p in papers] == reversed_ids


def test_sends_identity_and_query_params() -> None:
    handler, seen = canned()
    client = make_client(handler, email="me@example.org", api_key="k123")
    client.search(SearchQuery(raw="tau phosphorylation", max_results=7))
    esearch, efetch = (dict(r.url.params) for r in seen)
    assert esearch["db"] == "pubmed"
    assert esearch["tool"] == "test-tool"
    assert esearch["email"] == "me@example.org"
    assert esearch["api_key"] == "k123"
    assert esearch["term"] == "tau phosphorylation"
    assert esearch["retmax"] == "7"
    assert efetch["id"] == "24157548,28375731,25315507"
    assert efetch["retmode"] == "xml"


def test_omits_optional_identity_when_absent() -> None:
    handler, seen = canned()
    make_client(handler).search(SearchQuery(raw="q"))
    params = dict(seen[0].url.params)
    assert "email" not in params
    assert "api_key" not in params


def test_no_hits_returns_empty_without_calling_efetch() -> None:
    handler, seen = canned(esearch_body=ESEARCH_EMPTY)
    assert make_client(handler).search(SearchQuery(raw="zzqxv")) == []
    assert len(seen) == 1


# -- Rate limiting -------------------------------------------------------------


def test_paces_consecutive_requests_at_three_per_second_without_key() -> None:
    handler, _ = canned()
    clock = FakeClock()
    make_client(handler, clock=clock).search(SearchQuery(raw="q"))
    # esearch at t=0, efetch must wait the remaining 1/3 s.
    assert clock.sleeps == [pytest.approx(1 / 3)]


def test_paces_at_ten_per_second_with_key() -> None:
    handler, _ = canned()
    clock = FakeClock()
    make_client(handler, clock=clock, api_key="k").search(SearchQuery(raw="q"))
    assert clock.sleeps == [pytest.approx(0.1)]


# -- Retry ---------------------------------------------------------------------


def test_retries_transient_status_then_succeeds() -> None:
    handler, seen = flaky([503])
    clock = FakeClock()
    papers = make_client(handler, clock=clock).search(SearchQuery(raw="q"))
    assert len(papers) == 3
    assert len(seen) == 3  # failed esearch, esearch, efetch
    assert 0.5 in clock.sleeps  # the backoff before the retry


def test_retries_timeout_then_succeeds() -> None:
    handler, seen = flaky([httpx.ReadTimeout("slow")])
    papers = make_client(handler).search(SearchQuery(raw="q"))
    assert len(papers) == 3
    assert len(seen) == 3


@pytest.mark.parametrize("status", [429, 500, 503])
def test_gives_up_after_max_attempts_with_exponential_backoff(status: int) -> None:
    handler, seen = flaky([status, status, status])
    clock = FakeClock()
    with pytest.raises(LiteratureSourceUnavailableError, match=f"HTTP {status}.*3 attempts"):
        make_client(handler, clock=clock).search(SearchQuery(raw="q"))
    assert len(seen) == 3
    backoffs = [s for s in clock.sleeps if s in (0.5, 1.0)]
    assert backoffs == [0.5, 1.0]


def test_non_transient_status_is_not_retried() -> None:
    handler, seen = flaky([400, 400, 400])
    clock = FakeClock()
    with pytest.raises(LiteratureSourceUnavailableError, match="HTTP 400"):
        make_client(handler, clock=clock).search(SearchQuery(raw="q"))
    assert len(seen) == 1
    assert clock.sleeps == []


def test_connection_failure_is_retried_then_reported() -> None:
    errors: list[Exception | int] = [httpx.ConnectError("refused")] * 3
    handler, seen = flaky(errors)
    with pytest.raises(LiteratureSourceUnavailableError, match="could not be reached"):
        make_client(handler).search(SearchQuery(raw="q"))
    assert len(seen) == 3


def test_custom_retry_policy_is_honoured() -> None:
    handler, seen = flaky([503, 503])
    with pytest.raises(LiteratureSourceUnavailableError, match="1 attempts"):
        make_client(handler, retry=RetryPolicy(max_attempts=1)).search(SearchQuery(raw="q"))
    assert len(seen) == 1


# -- Parse errors are never retried --------------------------------------------


def test_malformed_esearch_is_parse_error() -> None:
    handler, seen = canned(esearch_body='{"unexpected": true}')
    with pytest.raises(LiteratureSourceParseError, match=r"esearchresult\.idlist"):
        make_client(handler).search(SearchQuery(raw="q"))
    assert len(seen) == 1


def test_malformed_efetch_is_parse_error() -> None:
    handler, seen = canned(efetch_body="<PubmedArticleSet><oops")
    with pytest.raises(LiteratureSourceParseError, match="not well-formed"):
        make_client(handler).search(SearchQuery(raw="q"))
    assert len(seen) == 2
