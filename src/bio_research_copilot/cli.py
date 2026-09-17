"""Command-line entry point.

This is the composition root: the one place that reads settings, builds real
adapters, and wires them into the application service. ``run`` accepts the
source as a parameter so tests can drive the full CLI with a fake provider.
"""

import argparse
import json
import sys
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from bio_research_copilot.adapters.literature import LiteratureSource, LiteratureSourceError
from bio_research_copilot.adapters.pubmed import PubMedClient
from bio_research_copilot.application import search_papers
from bio_research_copilot.config import Settings
from bio_research_copilot.domain import Paper

DISCLAIMER = "For research and educational use only. Not medical advice."

EXIT_OK = 0
EXIT_SOURCE_FAILED = 1
EXIT_BAD_INPUT = 2


def build_parser() -> argparse.ArgumentParser:
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


def run(argv: Sequence[str], source: LiteratureSource) -> int:
    """Execute the CLI against a given literature source. Returns the exit code."""
    args = build_parser().parse_args(argv)
    try:
        papers = search_papers(source, args.question, args.limit)
    except ValidationError as exc:
        reasons = "; ".join(str(err["msg"]) for err in exc.errors())
        print(f"error: invalid input: {reasons}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except LiteratureSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_SOURCE_FAILED

    output = render_json(papers) if args.format == "json" else render_table(papers)
    print(output)
    return EXIT_OK


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
        source = PubMedClient(
            http,
            tool=settings.ncbi_tool_name,
            email=settings.ncbi_contact_email,
            api_key=settings.ncbi_api_key,
        )
        return run(sys.argv[1:] if argv is None else argv, source)


if __name__ == "__main__":
    sys.exit(main())
