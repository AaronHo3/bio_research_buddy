"""Tests for the domain models.

These pin the contract that every adapter must satisfy and every downstream
component may rely on. Failures here mean the data model changed, which is a
deliberate decision, not a side effect.
"""

import pytest
from pydantic import ValidationError

from bio_research_copilot.domain import Paper, PublicationDate, SearchQuery, Source


def make_paper(**overrides: object) -> Paper:
    base: dict[str, object] = {
        "source": Source.PUBMED,
        "source_id": "12345678",
        "title": "CRISPR base editing off-target profiling",
        "abstract": "We profile off-target edits.",
        "authors": ["Smith J", "Doe A"],
        "journal": "Nature Biotechnology",
        "publication_date": PublicationDate(year=2023, month=4, day=12),
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    }
    base.update(overrides)
    return Paper.model_validate(base)


class TestPaper:
    def test_constructs_with_all_fields(self) -> None:
        paper = make_paper()
        assert paper.source_id == "12345678"
        assert paper.publication_date == PublicationDate(year=2023, month=4, day=12)

    def test_missing_optional_fields_are_none_not_empty(self) -> None:
        paper = make_paper(abstract=None, journal=None, publication_date=None, authors=[])
        assert paper.abstract is None
        assert paper.journal is None
        assert paper.publication_date is None
        assert paper.authors == []

    def test_title_is_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            make_paper(title="")
        with pytest.raises(ValidationError):
            make_paper(title="   ")

    def test_source_id_is_required_and_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            make_paper(source_id="")

    def test_url_must_be_http(self) -> None:
        with pytest.raises(ValidationError):
            make_paper(url="pubmed.ncbi.nlm.nih.gov/12345678/")

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_paper(source="arxiv")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_paper(pmid="12345678")

    def test_is_immutable(self) -> None:
        paper = make_paper()
        with pytest.raises(ValidationError):
            paper.title = "changed"

    def test_strips_surrounding_whitespace(self) -> None:
        paper = make_paper(title="  padded  ")
        assert paper.title == "padded"

    def test_json_round_trip_is_lossless_and_deterministic(self) -> None:
        paper = make_paper()
        dumped = paper.model_dump_json()
        assert Paper.model_validate_json(dumped) == paper
        assert paper.model_dump_json() == dumped


class TestPublicationDate:
    def test_year_only(self) -> None:
        date = PublicationDate(year=2021)
        assert (date.year, date.month, date.day) == (2021, None, None)

    def test_year_and_month(self) -> None:
        date = PublicationDate(year=2021, month=7)
        assert date.day is None

    def test_day_without_month_rejected(self) -> None:
        with pytest.raises(ValidationError, match="day cannot be set without month"):
            PublicationDate(year=2021, day=3)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("year", 999), ("year", 10000), ("month", 0), ("month", 13), ("day", 0), ("day", 32)],
    )
    def test_out_of_range_parts_rejected(self, field: str, value: int) -> None:
        payload: dict[str, int] = {"year": 2021, "month": 1, "day": 1}
        payload[field] = value
        with pytest.raises(ValidationError):
            PublicationDate.model_validate(payload)


class TestSearchQuery:
    def test_defaults(self) -> None:
        query = SearchQuery(raw="tau phosphorylation in Alzheimer's")
        assert query.max_results == 10

    def test_blank_question_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(raw="   ")

    @pytest.mark.parametrize("limit", [0, 101])
    def test_max_results_bounds(self, limit: int) -> None:
        with pytest.raises(ValidationError):
            SearchQuery(raw="q", max_results=limit)
