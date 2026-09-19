"""PubMed Central adapter: full text, where full text exists.

PubMed itself carries citations and abstracts and nothing more. Full text lives
in PubMed Central, and only a portion of it can be downloaded through the API,
so this adapter is built around absence being normal rather than exceptional.

Measured against real relevance-ranked queries, roughly half to two thirds of
PubMed results have a PMC record, and almost all of those return a usable body.
A typical body runs tens of thousands of characters against roughly fifteen
hundred for an abstract, so the papers that do resolve dominate the evidence
even when half the results do not.

Two consequences shape the interface. A caller asks for full text and receives a
mapping covering only the papers that had it, never an error for the ones that
did not; and every retrieved section is labelled, so a later stage can say which
claims rest on full text and which on an abstract alone.

Reference lists, tables, figure captions and supplementary blocks are dropped.
They are mostly citation strings and numbers, which would dilute a term-based
ranking without ever forming a sentence worth quoting.
"""

import re
import xml.etree.ElementTree as ET

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.literature import LiteratureSourceParseError

PMC_LINK_NAME = "pubmed_pmc"

#: Sections that carry no quotable prose. Matched case-insensitively against the
#: section title, so "References" and "Supplementary Material" are both dropped.
SKIPPED_SECTIONS = frozenset(
    (
        "references", "reference", "bibliography", "acknowledgements",
        "acknowledgments", "supplementary material", "supplementary materials",
        "supporting information", "conflict of interest", "conflicts of interest",
        "competing interests", "author contributions", "funding", "abbreviations",
        "data availability", "data availability statement", "ethics statement",
    )
)  # fmt: skip

_DROPPED_TAGS = frozenset(("ref-list", "table-wrap", "fig", "table", "supplementary-material"))

MIN_SECTION_CHARS = 120
"""Shorter than this and a section is a heading or a stub, not evidence."""


class FullTextSection(BaseModel):
    """One labelled span of a paper's body."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1, description='Section heading, or "body" when unlabelled.')
    text: str = Field(min_length=1)


class FullText(BaseModel):
    """A paper's retrievable body, in document order."""

    model_config = ConfigDict(frozen=True)

    pmid: str = Field(min_length=1)
    pmcid: str = Field(min_length=1)
    sections: list[FullTextSection] = Field(min_length=1)

    @property
    def characters(self) -> int:
        return sum(len(section.text) for section in self.sections)


def parse_pmc_links(payload: object) -> dict[str, str]:
    """PMID to PMCID, from an ``elink`` JSON response.

    The request must send one ``id`` parameter per PMID. Given a single
    comma-joined ``id``, elink answers with one linkset holding every link and
    no indication of which belongs to which paper, which is unusable here; given
    repeated parameters it answers with one linkset per input, which is not.

    Only the ``pubmed_pmc`` link type is read. ``pubmed_pmc_refs`` appears in the
    same response and means the opposite thing, papers in PMC citing this one,
    so taking it would attach a paper to text it did not write.

    Raises:
        LiteratureSourceParseError: the response was not the expected shape.
    """
    if not isinstance(payload, dict):
        msg = "elink response was not an object"
        raise LiteratureSourceParseError(msg)
    linksets = payload.get("linksets")
    if not isinstance(linksets, list):
        msg = "elink response did not contain linksets"
        raise LiteratureSourceParseError(msg)

    mapping: dict[str, str] = {}
    for entry in linksets:
        if not isinstance(entry, dict):
            continue
        ids = entry.get("ids")
        if not isinstance(ids, list) or len(ids) != 1:
            # A batched response cannot be attributed; skip rather than guess.
            continue
        for db in entry.get("linksetdbs") or []:
            if not isinstance(db, dict) or db.get("linkname") != PMC_LINK_NAME:
                continue
            links = [str(link) for link in db.get("links") or []]
            if links:
                mapping[str(ids[0])] = links[0]
    return mapping


def _clean(text: str) -> str:
    return " ".join(text.split())


def _section_text(element: ET.Element) -> str:
    """Prose inside an element, skipping references, tables and figures."""
    parts: list[str] = []
    for child in element:
        tag = child.tag.split("}")[-1]
        if tag in _DROPPED_TAGS or tag == "sec":
            continue
        if tag == "title":
            continue
        parts.append("".join(child.itertext()))
    return _clean(" ".join(parts))


def _walk(element: ET.Element, inherited: str, out: list[FullTextSection]) -> None:
    label = inherited
    title = element.find("./{*}title")
    if title is not None:
        text = _clean("".join(title.itertext()))
        if text:
            label = text

    if label.lower() in SKIPPED_SECTIONS:
        return

    body = _section_text(element)
    if len(body) >= MIN_SECTION_CHARS:
        out.append(FullTextSection(label=label or "body", text=body))

    for child in element.findall("./{*}sec"):
        _walk(child, label, out)


def parse_pmc_xml(xml_text: str) -> list[FullText]:
    """Full texts from a ``efetch db=pmc`` reply, skipping articles without a body.

    An article whose publisher blocks downloading comes back with metadata and
    no body. That is an ordinary outcome, not a failure, so it is skipped
    silently and the caller falls back to the abstract it already has.

    Raises:
        LiteratureSourceParseError: the payload was not parseable XML.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        msg = f"PMC response was not valid XML: {exc}"
        raise LiteratureSourceParseError(msg) from exc

    results: list[FullText] = []
    # iter() takes a literal tag and does not honour the {*} wildcard that
    # find() accepts, so articles are gathered with findall instead.
    articles = root.findall(".//{*}article") or root.findall(".//article")
    for article in articles:
        pmid = _article_id(article, "pmid")
        pmcid = _article_id(article, "pmcid") or _article_id(article, "pmcaid")
        body = article.find("./{*}body")
        # An Element with no children is falsy, so `or` on a find() result is a
        # trap here; identity is the only safe test.
        if body is None:
            body = article.find("./body")
        if pmid is None or pmcid is None or body is None:
            continue

        sections: list[FullTextSection] = []
        found_sec = False
        for sec in body.findall("./{*}sec"):
            found_sec = True
            _walk(sec, "", sections)
        if not found_sec:
            loose = _section_text(body)
            if len(loose) >= MIN_SECTION_CHARS:
                sections.append(FullTextSection(label="body", text=loose))
        if sections:
            results.append(FullText(pmid=pmid, pmcid=pmcid, sections=sections))
    return results


def _article_id(article: ET.Element, kind: str) -> str | None:
    for node in article.iter():
        if node.tag.split("}")[-1] != "article-id":
            continue
        if node.get("pub-id-type") == kind and node.text:
            # PMC ids appear as "PMC12190491"; the bare digits are what elink
            # and efetch both take, so the prefix is stripped here once.
            return re.sub(r"^PMC", "", node.text.strip())
    return None
