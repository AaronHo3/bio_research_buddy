"""Tests for the PubMed client's HTTP behaviour, fully offline.

``httpx.MockTransport`` lets a plain function stand in for the network. Each
test declares what PubMed "says" and asserts what the client does about it.
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
from bio_research_copilot.domain import SearchQuery

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
ESEARCH = (FIXTURES / "esearch_crispr.json").read_text(encoding="utf-8")
ESEARCH_EMPTY = (FIXTURES / "esearch_empty.json").read_text(encoding="utf-8")
EFETCH = (FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8")

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, **identity: str) -> PubMedClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return PubMedClient(http, tool="test-tool", **identity)


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


@pytest.mark.parametrize("status", [429, 500, 503])
def test_http_error_status_is_unavailable(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with pytest.raises(LiteratureSourceUnavailableError, match=f"HTTP {status}"):
        make_client(handler).search(SearchQuery(raw="q"))


def test_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LiteratureSourceUnavailableError, match="timed out"):
        make_client(handler).search(SearchQuery(raw="q"))


def test_connection_failure_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LiteratureSourceUnavailableError, match="could not be reached"):
        make_client(handler).search(SearchQuery(raw="q"))


def test_malformed_esearch_is_parse_error() -> None:
    handler, _ = canned(esearch_body='{"unexpected": true}')
    with pytest.raises(LiteratureSourceParseError, match=r"esearchresult\.idlist"):
        make_client(handler).search(SearchQuery(raw="q"))


def test_malformed_efetch_is_parse_error() -> None:
    handler, _ = canned(efetch_body="<PubmedArticleSet><oops")
    with pytest.raises(LiteratureSourceParseError, match="not well-formed"):
        make_client(handler).search(SearchQuery(raw="q"))
