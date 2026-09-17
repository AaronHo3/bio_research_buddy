"""End-to-end CLI tests with a fake literature source.

The golden file pins the exact bytes the JSON output must produce for the
recorded PubMed fixture. If a change alters output, the diff is the review.
"""

from pathlib import Path

import pytest

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.cli import EXIT_BAD_INPUT, EXIT_OK, EXIT_SOURCE_FAILED, run
from bio_research_copilot.domain import Paper, SearchQuery

FIXTURES = Path(__file__).parent.parent / "fixtures"
RECORDED_PAPERS = parse_efetch_xml((FIXTURES / "pubmed" / "efetch_crispr.xml").read_text("utf-8"))
GOLDEN_JSON = (FIXTURES / "cli" / "search_crispr.json").read_text(encoding="utf-8")


class StubSource:
    def __init__(self, papers: list[Paper] | None = None, error: Exception | None = None) -> None:
        self.papers = papers or []
        self.error = error
        self.queries: list[SearchQuery] = []

    def search(self, query: SearchQuery) -> list[Paper]:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.papers


def test_json_output_matches_golden_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        ["search", "CRISPR base editing off-target", "--limit", "3"], StubSource(RECORDED_PAPERS)
    )
    assert code == EXIT_OK
    assert capsys.readouterr().out == GOLDEN_JSON


def test_json_output_is_deterministic_across_runs(capsys: pytest.CaptureFixture[str]) -> None:
    source = StubSource(RECORDED_PAPERS)
    run(["search", "q"], source)
    first = capsys.readouterr().out
    run(["search", "q"], source)
    assert capsys.readouterr().out == first


def test_limit_and_question_reach_the_source() -> None:
    source = StubSource(RECORDED_PAPERS)
    run(["search", "tau phosphorylation", "--limit", "2"], source)
    assert source.queries == [SearchQuery(raw="tau phosphorylation", max_results=2)]


def test_table_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["search", "q", "--format", "table"], StubSource(RECORDED_PAPERS[:1]))
    assert code == EXIT_OK
    assert capsys.readouterr().out == (
        "1. [24157548] Genome engineering using the CRISPR-Cas9 system.\n"
        "   Nature protocols, 2013\n"
        "   https://pubmed.ncbi.nlm.nih.gov/24157548/\n"
    )


def test_no_results_json_is_empty_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["search", "zzqxv"], StubSource([])) == EXIT_OK
    assert capsys.readouterr().out == "[]\n"


def test_no_results_table_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    run(["search", "zzqxv", "--format", "table"], StubSource([]))
    assert capsys.readouterr().out == "No papers found.\n"


@pytest.mark.parametrize(
    "error",
    [
        LiteratureSourceUnavailableError("PubMed esearch.fcgi timed out"),
        LiteratureSourceParseError("bad xml"),
    ],
)
def test_source_failure_is_one_line_and_exit_1(
    error: Exception, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(["search", "q"], StubSource(error=error))
    captured = capsys.readouterr()
    assert code == EXIT_SOURCE_FAILED
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"
    assert "Traceback" not in captured.err


def test_blank_question_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    source = StubSource(RECORDED_PAPERS)
    code = run(["search", "   "], source)
    assert code == EXIT_BAD_INPUT
    assert capsys.readouterr().err.startswith("error: invalid input:")
    assert source.queries == []


def test_limit_out_of_range_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["search", "q", "--limit", "0"], StubSource()) == EXIT_BAD_INPUT
    assert "error: invalid input" in capsys.readouterr().err


def test_help_carries_disclaimer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run(["--help"], StubSource())
    assert exc.value.code == 0
    assert "Not medical advice" in capsys.readouterr().out
