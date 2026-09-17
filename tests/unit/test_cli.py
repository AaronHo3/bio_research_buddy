"""End-to-end CLI tests with a fake literature source and a temporary SQLite file.

The golden file pins the exact bytes the JSON search output must produce for
the recorded PubMed fixture. If a change alters output, the diff is the review.
"""

import io
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from bio_research_copilot.adapters.literature import (
    LiteratureSourceParseError,
    LiteratureSourceUnavailableError,
)
from bio_research_copilot.adapters.pubmed import parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.cli import (
    EXIT_BAD_INPUT,
    EXIT_DEPENDENCY_FAILED,
    EXIT_NOT_FOUND,
    EXIT_OK,
    Runtime,
    ensure_utf8_output,
    run,
)
from bio_research_copilot.domain import Paper, SearchQuery, Source

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


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "test.sqlite"


def runtime(source: StubSource, db: Path | None = None) -> Runtime:
    return Runtime(
        source=source,
        open_repository=SqliteRepository.open,
        default_db=db or Path("unused.sqlite"),
    )


# -- search --------------------------------------------------------------------


def test_json_output_matches_golden_file(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        ["search", "CRISPR base editing off-target", "--limit", "3"],
        runtime(StubSource(RECORDED_PAPERS)),
    )
    assert code == EXIT_OK
    assert capsys.readouterr().out == GOLDEN_JSON


def test_json_output_is_deterministic_across_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["search", "q"], rt)
    first = capsys.readouterr().out
    run(["search", "q"], rt)
    assert capsys.readouterr().out == first


def test_limit_and_question_reach_the_source() -> None:
    source = StubSource(RECORDED_PAPERS)
    run(["search", "tau phosphorylation", "--limit", "2"], runtime(source))
    assert source.queries == [SearchQuery(raw="tau phosphorylation", max_results=2)]


def test_table_output(capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["search", "q", "--format", "table"], runtime(StubSource(RECORDED_PAPERS[:1])))
    assert code == EXIT_OK
    assert capsys.readouterr().out == (
        "1. [24157548] Genome engineering using the CRISPR-Cas9 system.\n"
        "   Nature protocols, 2013\n"
        "   https://pubmed.ncbi.nlm.nih.gov/24157548/\n"
    )


def test_no_results_json_is_empty_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["search", "zzqxv"], runtime(StubSource([]))) == EXIT_OK
    assert capsys.readouterr().out == "[]\n"


def test_no_results_table_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    run(["search", "zzqxv", "--format", "table"], runtime(StubSource([])))
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
    code = run(["search", "q"], runtime(StubSource(error=error)))
    captured = capsys.readouterr()
    assert code == EXIT_DEPENDENCY_FAILED
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"
    assert "Traceback" not in captured.err


def test_blank_question_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    source = StubSource(RECORDED_PAPERS)
    code = run(["search", "   "], runtime(source))
    assert code == EXIT_BAD_INPUT
    assert capsys.readouterr().err.startswith("error: invalid input:")
    assert source.queries == []


def test_limit_out_of_range_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["search", "q", "--limit", "0"], runtime(StubSource())) == EXIT_BAD_INPUT
    assert "error: invalid input" in capsys.readouterr().err


def test_non_ascii_title_survives_a_legacy_console_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Windows consoles default to cp1252, which cannot encode a Greek beta."""
    paper = Paper(
        source=Source.PUBMED,
        source_id="1",
        title="Amyloid-β and tau in Alzheimer's disease",
        url="https://pubmed.ncbi.nlm.nih.gov/1/",
    )
    raw = io.BytesIO()
    legacy_stdout = io.TextIOWrapper(raw, encoding="cp1252", write_through=True)
    monkeypatch.setattr(sys, "stdout", legacy_stdout)

    ensure_utf8_output()
    code = run(["search", "q", "--format", "table"], runtime(StubSource([paper])))

    assert code == EXIT_OK
    assert "Amyloid-β" in raw.getvalue().decode("utf-8")


def test_help_carries_disclaimer(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        run(["--help"], runtime(StubSource()))
    assert exc.value.code == 0
    assert "Not medical advice" in capsys.readouterr().out


# -- ingest and show -----------------------------------------------------------


def test_ingest_stores_papers_and_prints_manifest(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        ["ingest", "CRISPR off-target", "--limit", "3", "--db", str(db)],
        runtime(StubSource(RECORDED_PAPERS)),
    )
    assert code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["question"] == "CRISPR off-target"
    assert report["max_results"] == 3
    assert report["papers_fetched"] == 3
    assert report["papers_inserted"] == 3
    assert report["papers_updated"] == 0
    assert report["paper_ids"] == ["24157548", "28375731", "25315507"]
    assert report["retrieval_id"] == 1
    assert datetime.fromisoformat(report["retrieved_at"]).tzinfo is not None

    with SqliteRepository.open(db) as repo:
        assert repo.count_papers() == 3


def test_second_ingest_reports_zero_inserted(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["ingest", "q", "--db", str(db)], rt)
    capsys.readouterr()
    run(["ingest", "q", "--db", str(db)], rt)
    report = json.loads(capsys.readouterr().out)
    assert report["papers_inserted"] == 0
    assert report["papers_updated"] == 3
    assert report["retrieval_id"] == 2


def test_ingest_uses_default_db_when_flag_omitted(db: Path) -> None:
    run(["ingest", "q"], runtime(StubSource(RECORDED_PAPERS), db=db))
    assert db.exists()


def test_ingest_source_failure_leaves_no_database_rows(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    error = LiteratureSourceUnavailableError("down")
    code = run(["ingest", "q", "--db", str(db)], runtime(StubSource(error=error)))
    assert code == EXIT_DEPENDENCY_FAILED
    assert capsys.readouterr().err == "error: down\n"
    with SqliteRepository.open(db) as repo:
        assert repo.count_papers() == 0
        assert repo.get_retrieval(1) is None


def test_show_prints_stored_paper(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["ingest", "q", "--db", str(db)], rt)
    capsys.readouterr()

    code = run(["show", "28375731", "--db", str(db)], rt)
    assert code == EXIT_OK
    shown = json.loads(capsys.readouterr().out)
    assert shown == RECORDED_PAPERS[1].model_dump(mode="json")


def test_show_unknown_id_is_exit_3(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["show", "0", "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_NOT_FOUND
    assert capsys.readouterr().err == "error: no stored pubmed paper with id 0\n"


def test_passages_prints_stored_chunks(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["ingest", "q", "--db", str(db)], rt)
    capsys.readouterr()

    code = run(["passages", "24157548", "--db", str(db)], rt)
    assert code == EXIT_OK
    shown = json.loads(capsys.readouterr().out)
    assert [p["chunk_id"] for p in shown] == ["pubmed:24157548:0", "pubmed:24157548:1"]
    assert shown[0]["section"] == "title"
    assert shown[1]["text"] == RECORDED_PAPERS[0].abstract


def test_passages_unknown_id_is_exit_3(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["passages", "0", "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_NOT_FOUND
    assert capsys.readouterr().err == "error: no stored passages for pubmed paper 0\n"


# -- query ---------------------------------------------------------------------


def test_query_ranks_stored_passages(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["ingest", "q", "--db", str(db)], rt)
    capsys.readouterr()

    code = run(["query", "double nicking off-target cleavage", "--k", "2", "--db", str(db)], rt)
    assert code == EXIT_OK
    evidence = json.loads(capsys.readouterr().out)
    assert len(evidence) == 2
    assert evidence[0]["rank"] == 1
    assert evidence[0]["method"] == "bm25-v1"
    assert evidence[0]["passage"]["source_id"] == "24157548"
    assert evidence[0]["score"] >= evidence[1]["score"]


def test_query_table_format(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rt = runtime(StubSource(RECORDED_PAPERS))
    run(["ingest", "q", "--db", str(db)], rt)
    capsys.readouterr()
    run(["query", "CRISPR", "--k", "1", "--format", "table", "--db", str(db)], rt)
    out = capsys.readouterr().out
    assert out.startswith("1. pubmed:")
    assert "score=" in out


def test_query_with_no_match_prints_empty_list(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(["query", "zebrafish", "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_OK
    assert capsys.readouterr().out == "[]\n"


def test_query_blank_question_is_exit_2(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["query", "  ", "--db", str(db)], runtime(StubSource())) == EXIT_BAD_INPUT
    assert capsys.readouterr().err.startswith("error: invalid input:")


# -- load-corpus, judge, evaluate -----------------------------------------------

BENCHMARK_DIR = Path(__file__).parent.parent.parent / "evaluation" / "benchmark_v1"


def test_load_corpus_stores_the_committed_benchmark(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        ["load-corpus", str(BENCHMARK_DIR / "corpus.xml"), "--db", str(db)], runtime(StubSource())
    )
    assert code == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["papers_fetched"] == 60
    assert report["papers_inserted"] == 60
    assert report["question"] == "corpus:corpus.xml"
    assert report["passages_written"] >= 119  # 60 titles + 59 abstracts, some structured


def test_load_corpus_missing_file_is_exit_2(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["load-corpus", "nope.xml", "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_BAD_INPUT
    assert capsys.readouterr().err.startswith("error: cannot read nope.xml")


def write_benchmark(path: Path, questions: list[dict[str, object]]) -> Path:
    """A throwaway benchmark file, so these tests never depend on judging progress."""
    path.write_text(
        json.dumps({"version": "t", "judgment_unit": "paper", "questions": questions}),
        encoding="utf-8",
    )
    return path


def test_judge_lists_only_unjudged_questions(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = write_benchmark(
        tmp_path / "questions.json",
        [
            {"id": "open", "topic": "t", "type": "mechanism", "question": "CRISPR off-target"},
            {
                "id": "done",
                "topic": "t",
                "type": "target",
                "question": "Cas9",
                "relevant": ["24157548"],
            },
            {"id": "una", "topic": "none", "type": "unanswerable", "question": "zebrafish"},
        ],
    )
    rt = runtime(StubSource())
    run(["load-corpus", str(BENCHMARK_DIR / "corpus.xml"), "--db", str(db)], rt)
    capsys.readouterr()

    argv = ["judge", str(bench), "--scope", "pool", "--pool", "3", "--db", str(db)]
    assert run(argv, rt) == EXIT_OK
    out = capsys.readouterr().out
    assert "## open [mechanism]" in out
    assert "## done" not in out  # already judged
    assert "## una" not in out  # unanswerable is judged by construction
    assert out.count("\n- ") <= 3


def test_judge_says_so_when_everything_is_judged(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = write_benchmark(
        tmp_path / "questions.json",
        [{"id": "una", "topic": "none", "type": "unanswerable", "question": "zebrafish"}],
    )
    rt = runtime(StubSource())
    run(["load-corpus", str(BENCHMARK_DIR / "corpus.xml"), "--db", str(db)], rt)
    capsys.readouterr()
    assert run(["judge", str(bench), "--scope", "pool", "--db", str(db)], rt) == EXIT_OK
    assert capsys.readouterr().out == "All questions are judged.\n"


def test_judge_topic_scope_lists_every_corpus_paper_with_markers(
    db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rt = runtime(StubSource())
    run(["load-corpus", str(BENCHMARK_DIR / "corpus.xml"), "--db", str(db)], rt)
    capsys.readouterr()

    assert run(["judge", str(BENCHMARK_DIR), "--db", str(db)], rt) == EXIT_OK  # topic is default
    out = capsys.readouterr().out

    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
    for topic, info in manifest["topics"].items():
        assert f"# {topic}" in out
        for pmid in info["pmids"]:
            assert pmid in out, f"{pmid} missing from the topic worksheet"
    assert "marker slots, in order: q01 q02 q03" in out
    # Judged papers carry an X in the slot of the question that lists them.
    benchmark = json.loads((BENCHMARK_DIR / "questions.json").read_text(encoding="utf-8"))
    q01 = next(q for q in benchmark["questions"] if q["id"] == "q01")
    for pmid in q01["relevant"]:
        assert "[X" in out and pmid in out


def test_judge_topic_scope_without_a_manifest_is_exit_2(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = write_benchmark(
        tmp_path / "questions.json",
        [{"id": "a", "topic": "t", "type": "mechanism", "question": "CRISPR"}],
    )
    code = run(["judge", str(bench), "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_BAD_INPUT
    assert "manifest.json" in capsys.readouterr().err


def test_evaluate_prints_summary_and_writes_report(
    db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rt = runtime(StubSource())
    run(["load-corpus", str(BENCHMARK_DIR / "corpus.xml"), "--db", str(db)], rt)
    capsys.readouterr()
    out_file = tmp_path / "results" / "report.json"
    code = run(
        ["evaluate", str(BENCHMARK_DIR), "--k", "5", "--out", str(out_file), "--db", str(db)], rt
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "benchmark benchmark_v1  retriever bm25-v1  k=5" in out
    assert "corpus: 60 papers" in out
    assert "unanswerable q10" in out
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["k"] == 5
    assert len(report["results"]) == 12


def test_evaluate_missing_benchmark_is_exit_2(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["evaluate", "nowhere", "--db", str(db)], runtime(StubSource()))
    assert code == EXIT_BAD_INPUT
    assert "cannot read nowhere" in capsys.readouterr().err


def test_show_unopenable_db_is_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    code = run(["show", "1", "--db", str(blocker / "db.sqlite")], runtime(StubSource()))
    assert code == EXIT_DEPENDENCY_FAILED
    assert capsys.readouterr().err.startswith("error: could not open database")
