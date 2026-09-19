"""PubMed adapter built on NCBI E-utilities.

Retrieval is a two-step flow:

1. ``esearch`` turns a free-text query into a ranked list of PMIDs.
2. ``efetch`` turns those PMIDs into full XML records.

This module owns all knowledge of the E-utilities URL scheme and the PubMed XML
schema. It exposes only domain objects and the errors declared in
:mod:`bio_research_copilot.adapters.literature`.

Usage policy: NCBI asks clients to identify themselves with ``tool`` and
``email`` and limits clients to 3 requests per second, or 10 with an API key.
The client paces itself to that limit and retries transient failures
(timeouts, connection errors, HTTP 429 and 5xx) with exponential backoff.
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.adapters.pmc import FullText, parse_pmc_links, parse_pmc_xml
from bio_research_copilot.adapters.resilience import MinIntervalLimiter, RetryPolicy
from bio_research_copilot.domain import Paper, PublicationDate, SearchQuery, Source

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
REQUESTS_PER_SECOND_WITHOUT_KEY = 3.0
REQUESTS_PER_SECOND_WITH_KEY = 10.0
_RETRYABLE_STATUS = frozenset({429}) | frozenset(range(500, 600))

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
        retry: Backoff schedule for transient failures. Defaults to 3 attempts.
        requests_per_second: Self-imposed pace. Defaults to NCBI's published
            limit for keyed or unkeyed clients.
        clock, sleep: Time functions, injectable so tests never really wait.
    """

    def __init__(
        self,
        http: httpx.Client,
        *,
        tool: str,
        email: str | None = None,
        api_key: str | None = None,
        retry: RetryPolicy | None = None,
        requests_per_second: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http
        self._identity: dict[str, str] = {"db": "pubmed", "tool": tool}
        if email:
            self._identity["email"] = email
        if api_key:
            self._identity["api_key"] = api_key
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        if requests_per_second is None:
            requests_per_second = (
                REQUESTS_PER_SECOND_WITH_KEY if api_key else REQUESTS_PER_SECOND_WITHOUT_KEY
            )
        self._limiter = MinIntervalLimiter(requests_per_second, clock=clock, sleep=sleep)

    def search(self, query: SearchQuery, term: str | None = None) -> list[Paper]:
        pmids = self._esearch(query, term)
        if not pmids:
            return []
        xml_text = self._efetch(pmids)
        papers = parse_efetch_xml(xml_text)
        # efetch does not guarantee esearch's relevance order; restore it.
        rank = {pmid: i for i, pmid in enumerate(pmids)}
        return sorted(papers, key=lambda p: rank.get(p.source_id, len(rank)))

    def fetch_full_text(self, pmids: Sequence[str], limit: int = 8) -> dict[str, FullText]:
        """Full text for up to ``limit`` of ``pmids``, preferring the order given.

        Availability is checked for every id before anything is downloaded.
        ``elink`` costs one request whatever the batch size, while a body runs to
        tens of thousands of characters, so learning first which papers have one
        means the expensive call is spent on papers that are both relevant and
        obtainable. Choosing the most relevant first and only then discovering
        most of them are not in PMC wastes the budget on the wrong papers.

        Absence is the normal case, not an error: only some papers have a PMC
        record, and only some of those permit download. A paper with no full
        text is simply missing from the result, and the caller keeps whatever
        abstract it already had.

        Raises:
            LiteratureSourceUnavailableError: NCBI could not be reached.
            LiteratureSourceParseError: NCBI answered in an unexpected shape.
        """
        wanted = [pmid for pmid in dict.fromkeys(pmids) if pmid]
        if not wanted or limit < 1:
            return {}

        pairs = [("dbfrom", "pubmed"), ("db", "pmc"), ("retmode", "json")]
        pairs += [("id", pmid) for pmid in wanted]
        try:
            payload = self._get_pairs("elink.fcgi", pairs).json()
        except ValueError as exc:
            msg = "elink response was not valid JSON"
            raise LiteratureSourceParseError(msg) from exc

        available = parse_pmc_links(payload)
        # Caller order is relevance order, so this keeps the most relevant of the
        # papers that actually have a body.
        chosen = [available[pmid] for pmid in wanted if pmid in available][:limit]
        if not chosen:
            return {}

        xml_text = self._get("efetch.fcgi", db="pmc", id=",".join(chosen), retmode="xml").text
        # The PMID inside the PMC record is authoritative; the elink mapping is
        # only used to decide what to ask for.
        return {text.pmid: text for text in parse_pmc_xml(xml_text) if text.pmid in wanted}

    # -- E-utilities calls ---------------------------------------------------

    def _esearch(self, query: SearchQuery, term: str | None = None) -> list[str]:
        """PMIDs for a question, falling back through progressively simpler queries.

        A planned query is tried first when one is supplied. If it matches
        nothing, the deterministic translation is tried, and then that stripped
        to its distinctive tokens. Each step is broader than the last, because a
        query returning nothing is the one failure the ranking stage cannot
        recover from.
        """
        attempts: list[str] = []
        if term:
            attempts.append(term)
        translated = to_search_term(query.raw)
        if translated not in attempts:
            attempts.append(translated)
        narrowed = narrow_search_term(translated)
        if narrowed and narrowed not in attempts:
            attempts.append(narrowed)

        ids: list[str] = []
        for attempt in attempts:
            ids = self._esearch_term(attempt, query.max_results)
            if ids:
                return ids
        return ids

    def _esearch_term(self, term: str, max_results: int) -> list[str]:
        payload = self._get(
            "esearch.fcgi",
            term=term,
            retmax=str(max_results),
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
        """One logical request: paced by the limiter, retried on transient failure."""
        return self._get_pairs(endpoint, list(params.items()))

    def _get_pairs(self, endpoint: str, pairs: list[tuple[str, str]]) -> httpx.Response:
        """As ``_get``, but allowing a parameter to repeat.

        ``elink`` needs one ``id`` parameter per record: given a single
        comma-joined value it answers with one undifferentiated linkset, which
        cannot be attributed back to the papers that produced it.
        """
        attempt = 0
        while True:
            attempt += 1
            self._limiter.wait()
            try:
                return self._request(endpoint, pairs)
            except _TransientError as exc:
                if attempt >= self._retry.max_attempts:
                    msg = f"{exc.message} (gave up after {attempt} attempts)"
                    raise LiteratureSourceUnavailableError(msg) from exc.cause
                self._sleep(self._retry.delay_before_retry(attempt))

    def _request(self, endpoint: str, params: list[tuple[str, str]]) -> httpx.Response:
        """A single HTTP call. Transient problems raise ``_TransientError``."""
        overridden = {key for key, _ in params}
        merged: list[tuple[str, str | int | float | bool | None]] = [
            (key, value) for key, value in self._identity.items() if key not in overridden
        ]
        merged.extend(params)
        try:
            response = self._http.get(f"{EUTILS_BASE_URL}/{endpoint}", params=merged)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            msg = f"PubMed {endpoint} returned HTTP {status}"
            if status in _RETRYABLE_STATUS:
                raise _TransientError(msg, exc) from exc
            raise LiteratureSourceUnavailableError(msg) from exc
        except httpx.TimeoutException as exc:
            raise _TransientError(f"PubMed {endpoint} timed out", exc) from exc
        except httpx.TransportError as exc:
            raise _TransientError(f"PubMed {endpoint} could not be reached: {exc}", exc) from exc
        return response


class _TransientError(Exception):
    """Internal signal: this request failed in a way worth retrying."""

    def __init__(self, message: str, cause: Exception) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


# -- XML translation ----------------------------------------------------------


QUESTION_WORDS = frozenset(
    (
        # Dropped before the term reaches esearch. NCBI ANDs every term it does
        # not recognise as a field, so one conversational word can take a query
        # from hundreds of papers to none: "Tell me about MYBPC3" returns
        # nothing, while "MYBPC3" returns several hundred. None of these carry
        # retrieval value, so removing them costs nothing.
        # Articles, conjunctions, prepositions, auxiliaries
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
        "could", "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "in", "into", "is", "it", "its", "may", "might", "must", "of", "on", "or",
        "should", "that", "the", "their", "them", "there", "these", "this",
        "those", "to", "was", "were", "what", "when", "where", "which", "who",
        "whom", "why", "will", "with", "would",
        # Pronouns and politeness: a question addressed to an assistant
        "i", "me", "my", "myself", "our", "please", "thanks", "us", "we", "you",
        "your",
        # Framing verbs and nouns: the request, not the subject
        "about", "describe", "detail", "details", "discuss", "explain",
        "explanation", "give", "information", "know", "learn", "let", "like",
        "list", "need", "outline", "overview", "provide", "question", "regarding",
        "summarise", "summarize", "summary", "tell", "understand", "want",
    )
)  # fmt: skip

#: Tokens worth keeping when a query has to be narrowed: gene symbols, protein
#: names and acronyms all carry a digit or an interior capital.
_DISTINCTIVE = re.compile(r"\d|(?<=.)[A-Z]")

NARROWED_TERM_TOKENS = 3
"""How many tokens a narrowed query keeps when nothing looks distinctive."""

_EDGE_PUNCTUATION = ".,;:!?()[]{}\"'"


def to_search_term(question: str) -> str:
    """A PubMed query built from a natural-language question.

    A question sent verbatim to esearch is not merely noisy, it can be actively
    wrong: NCBI's automatic term mapping reads "is" as an author name, and the
    resulting ``is[Author]`` clause is ANDed with everything else, so a question
    about a well-covered topic returns zero papers. Dropping question words
    removes that failure without touching the terms that carry meaning.

    Word forms are otherwise preserved, so "PD-1", "COVID-19" and "CRISPR/Cas9"
    reach PubMed intact; a tokeniser that split on punctuation would break them
    into parts that match far less precisely.

    Returns the original question when nothing would remain, because an empty
    term is worse than an imperfect one.
    """
    kept = [
        word
        for raw in question.split()
        if (word := raw.strip(_EDGE_PUNCTUATION)) and word.lower() not in QUESTION_WORDS
    ]
    return " ".join(kept) if kept else question.strip()


def narrow_search_term(term: str) -> str:
    """A shorter query to try when the first one matched nothing.

    An exhaustive list of words that must not be ANDed cannot be maintained, so
    a query that returns nothing is retried with only its most distinctive
    tokens. Gene symbols, protein names and acronyms all carry a digit or an
    interior capital, which is enough to pick "MYBPC3" out of "Tell me about
    MYBPC3" without knowing that "tell" was the problem. Failing that, the
    longest tokens are kept, since long words are rarely the framing ones.

    Returns an empty string when there is nothing left to narrow to, which the
    caller treats as no result rather than retrying.
    """
    tokens = term.split()
    if len(tokens) < 2:
        return ""
    distinctive = [token for token in tokens if _DISTINCTIVE.search(token)]
    if distinctive:
        return " ".join(distinctive)
    longest = sorted(tokens, key=len, reverse=True)[:NARROWED_TERM_TOKENS]
    if len(longest) == len(tokens):
        return ""
    # Restore the original order, so the query still reads as the question did.
    return " ".join(token for token in tokens if token in set(longest))


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
