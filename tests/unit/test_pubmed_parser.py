"""Tests for translating PubMed efetch XML into domain papers.

Two kinds of input: a recorded real response (shape drift protection) and
hand-built minimal records that exercise every missing-field branch.
"""

from pathlib import Path

import pytest

from bio_research_copilot.adapters.literature import LiteratureSourceParseError
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.domain import PublicationDate, Source

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pubmed"
RECORDED = (FIXTURES / "efetch_crispr.xml").read_text(encoding="utf-8")


def wrap(article_body: str, pmid: str = "1") -> str:
    return f"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">{pmid}</PMID>
      <Article PubModel="Print">
        {article_body}
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


MINIMAL = "<ArticleTitle>Minimal record.</ArticleTitle>"


class TestRecordedFixture:
    def test_parses_all_records(self) -> None:
        papers = parse_efetch_xml(RECORDED)
        assert [p.source_id for p in papers] == ["24157548", "28375731", "25315507"]

    def test_first_record_fields(self) -> None:
        paper = parse_efetch_xml(RECORDED)[0]
        assert paper.source is Source.PUBMED
        assert paper.title == "Genome engineering using the CRISPR-Cas9 system."
        assert paper.journal == "Nature protocols"
        assert paper.publication_date == PublicationDate(year=2013, month=11)
        assert paper.authors[:2] == ["Ran FA", "Hsu PD"]
        assert paper.abstract is not None
        assert paper.abstract.startswith("Targeted nucleases")
        assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/24157548/"

    def test_full_date_when_day_present(self) -> None:
        paper = parse_efetch_xml(RECORDED)[1]
        assert paper.publication_date == PublicationDate(year=2017, month=5, day=22)


class TestMissingAndOddFields:
    def test_minimal_record_yields_none_for_optional_fields(self) -> None:
        paper = parse_efetch_xml(wrap(MINIMAL, pmid="42"))[0]
        assert paper.source_id == "42"
        assert paper.abstract is None
        assert paper.journal is None
        assert paper.publication_date is None
        assert paper.authors == []

    def test_empty_abstract_element_is_none(self) -> None:
        body = MINIMAL + "<Abstract><AbstractText></AbstractText></Abstract>"
        assert parse_efetch_xml(wrap(body))[0].abstract is None

    def test_structured_abstract_keeps_labels(self) -> None:
        body = MINIMAL + (
            "<Abstract>"
            '<AbstractText Label="BACKGROUND">Context.</AbstractText>'
            '<AbstractText Label="RESULTS">Findings.</AbstractText>'
            "</Abstract>"
        )
        expected = "BACKGROUND: Context.\n\nRESULTS: Findings."
        assert parse_efetch_xml(wrap(body))[0].abstract == expected

    def test_inline_markup_in_title_is_flattened(self) -> None:
        body = (
            "<ArticleTitle>Role of <i>BRCA1</i> in H<sub>2</sub>O<sub>2</sub> response."
            "</ArticleTitle>"
        )
        assert parse_efetch_xml(wrap(body))[0].title == "Role of BRCA1 in H2O2 response."

    def test_collective_author_and_author_without_initials(self) -> None:
        body = MINIMAL + (
            "<AuthorList>"
            "<Author><CollectiveName>ENCODE Project Consortium</CollectiveName></Author>"
            "<Author><LastName>Okonkwo</LastName><ForeName>Ada</ForeName></Author>"
            "<Author><LastName>Solo</LastName></Author>"
            "</AuthorList>"
        )
        assert parse_efetch_xml(wrap(body))[0].authors == [
            "ENCODE Project Consortium",
            "Okonkwo Ada",
            "Solo",
        ]

    @pytest.mark.parametrize(
        ("pub_date_xml", "expected"),
        [
            ("<Year>2020</Year>", PublicationDate(year=2020)),
            ("<Year>2020</Year><Month>10</Month>", PublicationDate(year=2020, month=10)),
            (
                "<Year>2020</Year><Month>Oct</Month><Day>5</Day>",
                PublicationDate(year=2020, month=10, day=5),
            ),
            ("<Year>2020</Year><Month>Winter</Month><Day>5</Day>", PublicationDate(year=2020)),
            ("<MedlineDate>2012 Nov-Dec</MedlineDate>", PublicationDate(year=2012)),
            ("<MedlineDate>Spring 2012</MedlineDate>", None),
            ("", None),
        ],
    )
    def test_partial_and_medline_dates(
        self, pub_date_xml: str, expected: PublicationDate | None
    ) -> None:
        body = MINIMAL + (
            f"<Journal><JournalIssue><PubDate>{pub_date_xml}</PubDate></JournalIssue></Journal>"
        )
        assert parse_efetch_xml(wrap(body))[0].publication_date == expected


class TestMalformedInput:
    def test_not_xml_raises_parse_error(self) -> None:
        with pytest.raises(LiteratureSourceParseError, match="not well-formed"):
            parse_efetch_xml("<html>Service Unavailable</html><<")

    def test_missing_pmid_raises(self) -> None:
        xml = wrap(MINIMAL).replace('<PMID Version="1">1</PMID>', "")
        with pytest.raises(LiteratureSourceParseError, match="missing a PMID"):
            parse_efetch_xml(xml)

    def test_missing_title_raises(self) -> None:
        with pytest.raises(LiteratureSourceParseError, match="no ArticleTitle"):
            parse_efetch_xml(wrap("<Abstract><AbstractText>x</AbstractText></Abstract>", "7"))

    def test_empty_article_set_is_empty_list(self) -> None:
        assert parse_efetch_xml("<PubmedArticleSet></PubmedArticleSet>") == []
