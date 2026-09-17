"""PubMed adapter built on NCBI E-utilities.

Retrieval is a two-step flow:

1. ``esearch`` turns a free-text query into a ranked list of PMIDs.
2. ``efetch`` turns those PMIDs into full XML records.

This module owns all knowledge of the E-utilities URL scheme and the PubMed XML
schema. It exposes only domain objects and the errors declared in
:mod:`bio_research_copilot.adapters.literature`.

Usage policy: NCBI asks clients to identify themselves with ``tool`` and
``email`` and limits unkeyed clients to 3 requests per second. Retry, backoff,
and rate limiting are deliberately not implemented yet; see the project spec.
"""

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.domain import Paper, PublicationDate, SearchQuery, Source

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class PubMedClient:
    """Fetch normalized :class:`Paper` records from PubMed.

    Args:
        http: An ``httpx.Client``. Injected so tests can supply a mock transport
            and so the caller controls connection pooling and lifetime.
        tool: Client name sent to NCBI, per their usage policy.
        email: Contact address sent to NCBI. Optional but requested by NCBI.
        api_key: NCBI API key. Optional; raises the rate limit when present.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        tool: str,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._http = http
        self._identity: dict[str, str] = {"db": "pubmed", "tool": tool}
        if email:
            self._identity["email"] = email
        if api_key:
            self._identity["api_key"] = api_key

    def search(self, query: SearchQuery) -> list[Paper]:
        pmids = self._esearch(query)
        if not pmids:
            return []
        xml_text = self._efetch(pmids)
        papers = parse_efetch_xml(xml_text)
        # efetch does not guarantee esearch's relevance order; restore it.
        rank = {pmid: i for i, pmid in enumerate(pmids)}
        return sorted(papers, key=lambda p: rank.get(p.source_id, len(rank)))

    # -- E-utilities calls ---------------------------------------------------

    def _esearch(self, query: SearchQuery) -> list[str]:
        payload = self._get(
            "esearch.fcgi",
            term=query.raw,
            retmax=str(query.max_results),
            retmode="json",
            sort="relevance",
        )
        try:
            body: Any = payload.json()
            ids = body["esearchresult"]["idlist"]
        except (ValueError, KeyError, TypeError) as exc:
            msg = "esearch response did not contain esearchresult.idlist"
            raise LiteratureSourceParseError(msg) from exc
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            msg = "esearch idlist was not a list of strings"
            raise LiteratureSourceParseError(msg)
        return ids

    def _efetch(self, pmids: list[str]) -> str:
        return self._get("efetch.fcgi", id=",".join(pmids), retmode="xml").text

    def _get(self, endpoint: str, **params: str) -> httpx.Response:
        try:
            response = self._http.get(
                f"{EUTILS_BASE_URL}/{endpoint}", params={**self._identity, **params}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            msg = f"PubMed {endpoint} returned HTTP {exc.response.status_code}"
            raise LiteratureSourceUnavailableError(msg) from exc
        except httpx.TimeoutException as exc:
            msg = f"PubMed {endpoint} timed out"
            raise LiteratureSourceUnavailableError(msg) from exc
        except httpx.TransportError as exc:
            msg = f"PubMed {endpoint} could not be reached: {exc}"
            raise LiteratureSourceUnavailableError(msg) from exc
        return response


# -- XML translation ----------------------------------------------------------


def parse_efetch_xml(xml_text: str) -> list[Paper]:
    """Translate a ``PubmedArticleSet`` document into domain papers.

    Raises:
        LiteratureSourceParseError: on malformed XML or a record that lacks the
            fields a citable paper must have (PMID, title).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        msg = f"efetch response was not well-formed XML: {exc}"
        raise LiteratureSourceParseError(msg) from exc
    return [_parse_article(node) for node in root.iter("PubmedArticle")]


def _parse_article(node: ET.Element) -> Paper:
    citation = node.find("MedlineCitation")
    article = citation.find("Article") if citation is not None else None
    if citation is None or article is None:
        msg = "PubmedArticle is missing MedlineCitation/Article"
        raise LiteratureSourceParseError(msg)

    pmid = _text(citation.find("PMID"))
    if not pmid:
        msg = "PubmedArticle is missing a PMID"
        raise LiteratureSourceParseError(msg)

    title = _text(article.find("ArticleTitle"))
    if not title:
        msg = f"PMID {pmid} has no ArticleTitle"
        raise LiteratureSourceParseError(msg)

    return Paper(
        source=Source.PUBMED,
        source_id=pmid,
        title=title,
        abstract=_parse_abstract(article.find("Abstract")),
        authors=[a for a in map(_parse_author, article.findall("AuthorList/Author")) if a],
        journal=_text(article.find("Journal/Title")) or None,
        publication_date=_parse_pub_date(article.find("Journal/JournalIssue/PubDate")),
        url=PUBMED_ARTICLE_URL.format(pmid=pmid),
    )


def _text(element: ET.Element | None) -> str:
    """All text under an element, flattened. Titles and abstracts may hold inline markup."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _parse_abstract(abstract: ET.Element | None) -> str | None:
    if abstract is None:
        return None
    sections: list[str] = []
    for part in abstract.findall("AbstractText"):
        body = _text(part)
        if not body:
            continue
        label = part.get("Label")
        sections.append(f"{label}: {body}" if label else body)
    return "\n\n".join(sections) or None


def _parse_author(author: ET.Element) -> str:
    collective = _text(author.find("CollectiveName"))
    if collective:
        return collective
    last = _text(author.find("LastName"))
    initials = _text(author.find("Initials")) or _text(author.find("ForeName"))
    return f"{last} {initials}".strip()


def _parse_pub_date(pub_date: ET.Element | None) -> PublicationDate | None:
    """Read ``<PubDate>``, which holds Year/Month/Day or a free-text MedlineDate.

    Precision is never invented: an unparseable month drops the month and day;
    a MedlineDate like ``2012 Nov-Dec`` keeps only the year.
    """
    if pub_date is None:
        return None
    year = _parse_int(_text(pub_date.find("Year")))
    if year is None:
        medline = _text(pub_date.find("MedlineDate"))
        year = _parse_int(medline[:4])
        return PublicationDate(year=year) if year else None
    month = _parse_month(_text(pub_date.find("Month")))
    day = _parse_int(_text(pub_date.find("Day"))) if month else None
    try:
        return PublicationDate(year=year, month=month, day=day)
    except ValueError:
        return PublicationDate(year=year)


def _parse_month(raw: str) -> int | None:
    if not raw:
        return None
    if raw.isdigit():
        value = int(raw)
        return value if 1 <= value <= 12 else None
    return _MONTHS.get(raw[:3].lower())


def _parse_int(raw: str) -> int | None:
    return int(raw) if raw.isdigit() else None
