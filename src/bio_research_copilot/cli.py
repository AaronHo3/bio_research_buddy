"""Command-line entry point.

This is the composition root: the one place that reads settings, builds real
adapters, and wires them into the application services. ``run`` receives a
``Runtime`` bundle so tests can drive the full CLI with a fake source and a
throwaway database.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from bio_research_copilot.adapters.literature import LiteratureSource, LiteratureSourceError
from bio_research_copilot.adapters.persistence import PaperRepository, PersistenceError
from bio_research_copilot.adapters.pubmed import PubMedClient
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import ingest_papers, search_papers
from bio_research_copilot.config import Settings
from bio_research_copilot.domain import Paper, Source

DISCLAIMER = "For research and educational use only. Not medical advice."

EXIT_OK = 0
EXIT_DEPENDENCY_FAILED = 1  # literature source or database failed
EXIT_BAD_INPUT = 2
EXIT_NOT_FOUND = 3


@dataclass(frozen=True)
class Runtime:
    """Everything the commands need from the outside world."""

    source: LiteratureSource
    open_repository: Callable[[Path], AbstractContextManager[PaperRepository]]
    default_db: Path


def build_parser(default_db: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bio-copilot",
        description="Evidence-grounded biomedical literature assistant.",
        epilog=DISCLAIMER,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Find papers for a research question.")
    search.add_argument("question", help="The research question, in plain language.")
    search.add_argument(
        "--limit", type=int, default=5, help="Maximum number of papers (default: 5)."
    )
    search.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output format (default: json).",
    )

    ingest = subparsers.add_parser(
        "ingest", help="Search, store the papers locally, and record the run."
    )
    ingest.add_argument("question", help="The research question, in plain language.")
    ingest.add_argument(
        "--limit", type=int, default=5, help="Maximum number of papers (default: 5)."
    )
    ingest.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    show = subparsers.add_parser("show", help="Print one stored paper.")
    show.add_argument("source_id", help="The paper's identifier within its source, e.g. a PMID.")
    show.add_argument(
        "--source",
        choices=[s.value for s in Source],
        default=Source.PUBMED.value,
        help="Literature source (default: pubmed).",
    )
    show.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    passages = subparsers.add_parser("passages", help="Print the stored passages of one paper.")
    passages.add_argument("source_id", help="The paper's identifier within its source.")
    passages.add_argument(
        "--source",
        choices=[s.value for s in Source],
        default=Source.PUBMED.value,
        help="Literature source (default: pubmed).",
    )
    passages.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )
    return parser


def render_json(papers: list[Paper]) -> str:
    """Stable, human-readable JSON. Field order follows the model definition."""
    return json.dumps(
        [paper.model_dump(mode="json") for paper in papers], indent=2, ensure_ascii=False
    )


def render_table(papers: list[Paper]) -> str:
    if not papers:
        return "No papers found."
    lines: list[str] = []
    for rank, paper in enumerate(papers, start=1):
        year = str(paper.publication_date.year) if paper.publication_date else "n.d."
        journal = paper.journal or "unknown journal"
        lines.append(f"{rank}. [{paper.source_id}] {paper.title}")
        lines.append(f"   {journal}, {year}")
        lines.append(f"   {paper.url}")
    return "\n".join(lines)


def _cmd_search(args: argparse.Namespace, runtime: Runtime) -> int:
    papers = search_papers(runtime.source, args.question, args.limit)
    print(render_json(papers) if args.format == "json" else render_table(papers))
    return EXIT_OK


def _cmd_ingest(args: argparse.Namespace, runtime: Runtime) -> int:
    with runtime.open_repository(args.db) as repository:
        report = ingest_papers(runtime.source, repository, args.question, args.limit)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return EXIT_OK


def _cmd_show(args: argparse.Namespace, runtime: Runtime) -> int:
    with runtime.open_repository(args.db) as repository:
        paper = repository.get_paper(Source(args.source), args.source_id)
    if paper is None:
        print(f"error: no stored {args.source} paper with id {args.source_id}", file=sys.stderr)
        return EXIT_NOT_FOUND
    print(json.dumps(paper.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return EXIT_OK


def _cmd_passages(args: argparse.Namespace, runtime: Runtime) -> int:
    with runtime.open_repository(args.db) as repository:
        passages = repository.get_passages(Source(args.source), args.source_id)
    if not passages:
        print(
            f"error: no stored passages for {args.source} paper {args.source_id}", file=sys.stderr
        )
        return EXIT_NOT_FOUND
    print(json.dumps([p.model_dump(mode="json") for p in passages], indent=2, ensure_ascii=False))
    return EXIT_OK


_COMMANDS: dict[str, Callable[[argparse.Namespace, Runtime], int]] = {
    "search": _cmd_search,
    "ingest": _cmd_ingest,
    "show": _cmd_show,
    "passages": _cmd_passages,
}


def run(argv: Sequence[str], runtime: Runtime) -> int:
    """Execute the CLI against the given runtime. Returns the exit code."""
    args = build_parser(runtime.default_db).parse_args(argv)
    try:
        return _COMMANDS[args.command](args, runtime)
    except ValidationError as exc:
        reasons = "; ".join(str(err["msg"]) for err in exc.errors())
        print(f"error: invalid input: {reasons}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except (LiteratureSourceError, PersistenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DEPENDENCY_FAILED


def ensure_utf8_output() -> None:
    """Force UTF-8 on stdout/stderr.

    Windows consoles default Python to a legacy code page (cp1252), which cannot
    encode characters common in biomedical titles such as Greek letters. Without
    this, printing a title containing a beta crashes with UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ensure_utf8_output()
    settings = Settings()
    with httpx.Client(timeout=settings.http_timeout_seconds) as http:
        runtime = Runtime(
            source=PubMedClient(
                http,
                tool=settings.ncbi_tool_name,
                email=settings.ncbi_contact_email,
                api_key=settings.ncbi_api_key,
            ),
            open_repository=SqliteRepository.open,
            default_db=settings.database_path,
        )
        return run(sys.argv[1:] if argv is None else argv, runtime)


if __name__ == "__main__":
    sys.exit(main())
