"""One real call to NCBI. Deselected by default; run with ``pytest -m live``."""

import os

import httpx
import pytest

from bio_research_copilot.adapters.pubmed import PubMedClient
from bio_research_copilot.domain import SearchQuery, Source

pytestmark = pytest.mark.live


def test_real_pubmed_search_returns_normalized_papers() -> None:
    with httpx.Client(timeout=30) as http:
        client = PubMedClient(
            http,
            tool=os.environ.get("NCBI_TOOL_NAME", "bio-research-copilot"),
            email=os.environ.get("NCBI_CONTACT_EMAIL") or None,
            api_key=os.environ.get("NCBI_API_KEY") or None,
        )
        papers = client.search(SearchQuery(raw="CRISPR base editing off-target", max_results=3))
    assert len(papers) == 3
    for paper in papers:
        assert paper.source is Source.PUBMED
        assert paper.source_id.isdigit()
        assert paper.title
        assert paper.url == f"https://pubmed.ncbi.nlm.nih.gov/{paper.source_id}/"
