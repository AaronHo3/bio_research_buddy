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

from bio_research_copilot.adapters.embedding import EmbeddingError, EmbeddingModel
from bio_research_copilot.adapters.fastembed_model import DEFAULT_MODEL, FastEmbedModel
from bio_research_copilot.adapters.literature import LiteratureSource, LiteratureSourceError
from bio_research_copilot.adapters.persistence import PaperRepository, PersistenceError
from bio_research_copilot.adapters.pubmed import PubMedClient, parse_efetch_xml
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import (
    EvaluationReport,
    embed_corpus,
    evaluate_retrieval,
    ingest_papers,
    load_corpus,
    query_passages,
    search_papers,
)
from bio_research_copilot.application.evaluate import collapse_to_papers
from bio_research_copilot.application.query import (
    RetrieverFactory,
    build_bm25,
    build_dense,
    build_hybrid,
)
from bio_research_copilot.config import Settings
from bio_research_copilot.domain import Paper, RetrievedEvidence, SearchQuery, Source
from bio_research_copilot.evaluation import (
    Benchmark,
    CorpusManifest,
    load_benchmark,
    load_manifest,
)
from bio_research_copilot.retrieval import BM25Index

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
    build_embedding_model: Callable[[str], EmbeddingModel] = FastEmbedModel
    default_embedding_model: str = DEFAULT_MODEL


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

    query = subparsers.add_parser(
        "query", help="Rank stored passages for a question (lexical BM25, no network)."
    )
    query.add_argument("question", help="The research question, in plain language.")
    query.add_argument("--k", type=int, default=5, help="Number of passages (default: 5).")
    query.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval method (default: bm25, the lexical baseline).",
    )
    query.add_argument("--model", default=None, help="Embedding model, for dense and hybrid.")
    query.add_argument(
        "--format",
        choices=("json", "table"),
        default="json",
        help="Output format (default: json).",
    )
    query.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    load = subparsers.add_parser(
        "load-corpus", help="Store papers from a saved PubMed efetch XML file (no network)."
    )
    load.add_argument("xml_path", type=Path, help="Path to a PubmedArticleSet XML file.")
    load.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    judge = subparsers.add_parser("judge", help="Print candidate papers to judge for relevance.")
    judge.add_argument("benchmark", type=Path, help="Benchmark directory or questions.json.")
    judge.add_argument(
        "--scope",
        choices=("topic", "pool"),
        default="topic",
        help="topic: every corpus paper grouped by topic, exhaustive (default). "
        "pool: only the retriever's top candidates for unjudged questions.",
    )
    judge.add_argument("--pool", type=int, default=10, help="Candidates per question, pool scope.")
    judge.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    embed = subparsers.add_parser(
        "embed", help="Compute and store vectors for stored passages (incremental)."
    )
    embed.add_argument("--model", default=None, help=f"Embedding model (default: {DEFAULT_MODEL}).")
    embed.add_argument("--batch-size", type=int, default=64, help="Passages per batch.")
    embed.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Score the lexical retriever against a benchmark and print a summary."
    )
    evaluate.add_argument("benchmark", type=Path, help="Benchmark directory or questions.json.")
    evaluate.add_argument("--k", type=int, default=10, help="Cut-off for metrics (default: 10).")
    evaluate.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval method to score (default: bm25).",
    )
    evaluate.add_argument("--model", default=None, help="Embedding model, for dense and hybrid.")
    evaluate.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    evaluate.add_argument(
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


def render_evidence_table(evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return "No matching passages."
    lines: list[str] = []
    for item in evidence:
        p = item.passage
        snippet = p.text if len(p.text) <= 120 else p.text[:117] + "..."
        lines.append(f"{item.rank}. {p.chunk_id}  score={item.score:.3f}  [{p.section}]")
        lines.append(f"   {snippet}")
    return "\n".join(lines)


def _build_retriever_factory(args: argparse.Namespace, runtime: Runtime) -> RetrieverFactory:
    """Turn --method and --model into the factory the use cases expect."""
    if args.method == "bm25":
        return build_bm25
    model = runtime.build_embedding_model(args.model or runtime.default_embedding_model)
    if args.method == "hybrid":
        return lambda repository: build_hybrid(repository, model)
    return lambda repository: build_dense(repository, model)


def _cmd_query(args: argparse.Namespace, runtime: Runtime) -> int:
    build = _build_retriever_factory(args, runtime)
    with runtime.open_repository(args.db) as repository:
        evidence = query_passages(repository, args.question, args.k, build_retriever=build)
    if args.format == "table":
        print(render_evidence_table(evidence))
    else:
        print(
            json.dumps([e.model_dump(mode="json") for e in evidence], indent=2, ensure_ascii=False)
        )
    return EXIT_OK


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {path}: {exc.strerror or exc}", file=sys.stderr)
        return None


def _cmd_load_corpus(args: argparse.Namespace, runtime: Runtime) -> int:
    xml_text = _read_text(args.xml_path)
    if xml_text is None:
        return EXIT_BAD_INPUT
    papers = parse_efetch_xml(xml_text)
    with runtime.open_repository(args.db) as repository:
        report = load_corpus(repository, papers, label=args.xml_path.name)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return EXIT_OK


def _snippet(text: str | None, limit: int) -> str:
    flat = (text or "").replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def render_pool_worksheet(benchmark: Benchmark, repository: PaperRepository, pool: int) -> str:
    """Top candidates per unjudged question, as ranked by the lexical retriever.

    Cheap, but it only ever shows what this retriever already finds, so recall
    measured against these judgments is biased upward. Prefer the topic scope
    when the corpus is small enough to judge exhaustively.
    """
    index = BM25Index(repository.all_passages())
    lines: list[str] = []
    for q in benchmark.questions:
        if q.judged:
            continue
        lines.append(f"## {q.id} [{q.type}] {q.question}")
        evidence = index.search(SearchQuery(raw=q.question, max_results=100))
        for pid in collapse_to_papers(evidence, pool):
            paper = repository.get_paper(Source.PUBMED, pid)
            if paper is None:
                continue
            lines.append(f"- {pid}  {paper.title}")
            lines.append(f"    {_snippet(paper.abstract, 240)}")
        lines.append("")
    return "\n".join(lines).rstrip() if lines else "All questions are judged."


def render_topic_worksheet(
    benchmark: Benchmark, manifest: CorpusManifest, repository: PaperRepository
) -> str:
    """Every corpus paper grouped by topic, with its current judgments marked.

    Exhaustive judging: each topic's papers are read once and judged against all
    of that topic's questions, so recall is exact rather than pooled. The marker
    shows the current state, one slot per question in the order listed above it.
    """
    lines: list[str] = []
    for topic, snapshot in manifest.topics.items():
        questions = [q for q in benchmark.questions if q.topic == topic]
        if not questions:
            continue
        slots = " ".join(q.id for q in questions)
        lines.append(f"# {topic}  ({len(snapshot.pmids)} papers)")
        lines.append(f"  query: {snapshot.query}")
        for q in questions:
            lines.append(f"  {q.id} [{q.type}] {q.question}")
        lines.append(f"  marker slots, in order: {slots}")
        lines.append("")
        for pmid in snapshot.pmids:
            paper = repository.get_paper(Source.PUBMED, pmid)
            if paper is None:
                lines.append(f"[???] {pmid}  NOT IN STORE")
                continue
            marker = "".join("X" if pmid in q.relevant else "." for q in questions)
            lines.append(f"[{marker}] {pmid}  {paper.title}")
            lines.append(f"      {_snippet(paper.abstract, 320)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _cmd_judge(args: argparse.Namespace, runtime: Runtime) -> int:
    if not args.benchmark.exists():
        print(f"error: cannot read {args.benchmark}: not found", file=sys.stderr)
        return EXIT_BAD_INPUT
    benchmark = load_benchmark(args.benchmark)
    manifest = None
    if args.scope == "topic":
        manifest_dir = args.benchmark if args.benchmark.is_dir() else args.benchmark.parent
        if not (manifest_dir / "manifest.json").exists():
            print(
                f"error: cannot read {manifest_dir / 'manifest.json'}: not found", file=sys.stderr
            )
            return EXIT_BAD_INPUT
        manifest = load_manifest(manifest_dir)
    with runtime.open_repository(args.db) as repository:
        if manifest is not None:
            text = render_topic_worksheet(benchmark, manifest, repository)
        else:
            text = render_pool_worksheet(benchmark, repository, args.pool)
    print(text)
    return EXIT_OK


def render_evaluation_summary(report: EvaluationReport) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    lines = [
        f"benchmark {report.benchmark_version}  retriever {report.retriever}  k={report.k}",
        f"corpus: {report.corpus_papers} papers, {report.corpus_passages} passages",
        f"judged questions: {report.judged_questions}"
        + (
            f"  (unjudged: {', '.join(report.unjudged_questions)})"
            if report.unjudged_questions
            else ""
        ),
        f"mean  P@{report.k}={fmt(report.mean_precision)}  R@{report.k}={fmt(report.mean_recall)}"
        f"  MRR={fmt(report.mean_mrr)}  nDCG@{report.k}={fmt(report.mean_ndcg)}"
        f"  latency={report.mean_latency_ms:.1f}ms",
    ]
    for qtype, s in report.by_type.items():
        lines.append(
            f"  {qtype:<12} n={s.questions}  P={s.precision:.3f}  R={s.recall:.3f}"
            f"  MRR={s.mrr:.3f}  nDCG={s.ndcg:.3f}"
        )
    for u in report.unanswerable:
        top = "none" if u.top_score is None else f"{u.top_score:.3f}"
        lines.append(f"  unanswerable {u.id}: {u.papers_returned} papers returned, top score {top}")
    return "\n".join(lines)


def _cmd_evaluate(args: argparse.Namespace, runtime: Runtime) -> int:
    if not args.benchmark.exists():
        print(f"error: cannot read {args.benchmark}: not found", file=sys.stderr)
        return EXIT_BAD_INPUT
    benchmark = load_benchmark(args.benchmark)
    build = _build_retriever_factory(args, runtime)
    with runtime.open_repository(args.db) as repository:
        report = evaluate_retrieval(repository, benchmark, k=args.k, build_retriever=build)
    print(render_evaluation_summary(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report written to {args.out}")
    return EXIT_OK


def _cmd_embed(args: argparse.Namespace, runtime: Runtime) -> int:
    model = runtime.build_embedding_model(args.model or runtime.default_embedding_model)
    with runtime.open_repository(args.db) as repository:
        report = embed_corpus(repository, model, batch_size=args.batch_size)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return EXIT_OK


_COMMANDS: dict[str, Callable[[argparse.Namespace, Runtime], int]] = {
    "search": _cmd_search,
    "ingest": _cmd_ingest,
    "show": _cmd_show,
    "passages": _cmd_passages,
    "query": _cmd_query,
    "load-corpus": _cmd_load_corpus,
    "judge": _cmd_judge,
    "embed": _cmd_embed,
    "evaluate": _cmd_evaluate,
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
    except (LiteratureSourceError, PersistenceError, EmbeddingError) as exc:
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
            build_embedding_model=lambda name: FastEmbedModel(
                name, cache_dir=settings.embedding_cache_dir
            ),
            default_embedding_model=settings.embedding_model,
        )
        return run(sys.argv[1:] if argv is None else argv, runtime)


if __name__ == "__main__":
    sys.exit(main())
