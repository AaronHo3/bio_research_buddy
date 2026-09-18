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
from bio_research_copilot.adapters.fastembed_model import DEFAULT_MODEL
from bio_research_copilot.adapters.fastembed_reranker import DEFAULT_RERANK_MODEL
from bio_research_copilot.adapters.literature import LiteratureSource, LiteratureSourceError
from bio_research_copilot.adapters.model_factory import (
    build_embedding_model,
    build_reranker,
)
from bio_research_copilot.adapters.persistence import PaperRepository, PersistenceError
from bio_research_copilot.adapters.pubmed import PubMedClient, parse_efetch_xml
from bio_research_copilot.adapters.reranking import Reranker, RerankerError
from bio_research_copilot.adapters.sqlite_repository import SqliteRepository
from bio_research_copilot.application import (
    EvaluationReport,
    answer_question,
    embed_corpus,
    evaluate_retrieval,
    ingest_papers,
    load_corpus,
    query_passages,
    search_papers,
)
from bio_research_copilot.application.evaluate import collapse_to_papers
from bio_research_copilot.application.evaluate_synthesis import (
    SynthesisReport,
    evaluate_synthesis,
)
from bio_research_copilot.application.judge_faithfulness import (
    FaithfulnessReport,
    build_worksheet,
    judge_report,
)
from bio_research_copilot.application.query import (
    RetrieverFactory,
    build_bm25,
    build_dense,
    build_hybrid,
    with_reranking,
)
from bio_research_copilot.config import Settings
from bio_research_copilot.domain import (
    Paper,
    ResearchAnswer,
    RetrievedEvidence,
    SearchQuery,
    Source,
)
from bio_research_copilot.evaluation import (
    Benchmark,
    CorpusManifest,
    load_benchmark,
    load_manifest,
)
from bio_research_copilot.retrieval import BM25Index
from bio_research_copilot.synthesis import (
    DEFAULT_CHAR_BUDGET,
    DEFAULT_LLM_MODEL,
    SynthesisError,
    SynthesisProvider,
    build_llm_synthesizer,
)
from bio_research_copilot.synthesis.faithfulness import (
    DEFAULT_JUDGE_MODEL,
    FaithfulnessJudge,
    LlmFaithfulnessJudge,
    build_judge,
)
from bio_research_copilot.synthesis.llm import LlmSynthesizer

NEWLINE = chr(10)

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
    build_embedding_model: Callable[[str], EmbeddingModel] = build_embedding_model
    default_embedding_model: str = DEFAULT_MODEL
    build_reranker: Callable[[str], Reranker] = build_reranker
    default_rerank_model: str = DEFAULT_RERANK_MODEL
    build_synthesizer: Callable[[str], SynthesisProvider] = build_llm_synthesizer
    default_llm_model: str = DEFAULT_LLM_MODEL
    build_judge: Callable[[str], FaithfulnessJudge] = build_judge
    default_judge_model: str = DEFAULT_JUDGE_MODEL


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
        "--rerank",
        action="store_true",
        help="Reorder the shortlist with a cross-encoder (slower, usually better ordering).",
    )
    query.add_argument("--rerank-model", default=None, help="Cross-encoder model, for --rerank.")
    query.add_argument(
        "--rerank-pool", type=int, default=25, help="Candidates to rerank (default: 25)."
    )
    query.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    answer = subparsers.add_parser(
        "answer", help="Answer a question from stored evidence, with every claim cited."
    )
    answer.add_argument("question", help="The research question, in plain language.")
    answer.add_argument(
        "--k", type=int, default=8, help="Passages to retrieve as evidence (default: 8)."
    )
    answer.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval method (default: bm25).",
    )
    answer.add_argument("--model", default=None, help="Embedding model, for dense and hybrid.")
    answer.add_argument(
        "--rerank",
        action="store_true",
        help="Reorder the shortlist with a cross-encoder before answering.",
    )
    answer.add_argument("--rerank-model", default=None, help="Cross-encoder model, for --rerank.")
    answer.add_argument(
        "--rerank-pool", type=int, default=25, help="Candidates to rerank (default: 25)."
    )
    answer.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Abstain when the top passage scores below this. Scales differ by method.",
    )
    answer.add_argument(
        "--char-budget",
        type=int,
        default=DEFAULT_CHAR_BUDGET,
        help=f"Evidence characters given to synthesis (default: {DEFAULT_CHAR_BUDGET}).",
    )
    answer.add_argument(
        "--synthesis",
        choices=("extractive", "llm"),
        default="extractive",
        help="How claims are written (default: extractive, which needs no API key).",
    )
    answer.add_argument("--llm-model", default=None, help="Model name, for --synthesis llm.")
    answer.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format (default: text).",
    )
    answer.add_argument(
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
    evaluate.add_argument(
        "--rerank",
        action="store_true",
        help="Reorder the shortlist with a cross-encoder (slower, usually better ordering).",
    )
    evaluate.add_argument("--rerank-model", default=None, help="Cross-encoder model, for --rerank.")
    evaluate.add_argument(
        "--rerank-pool", type=int, default=25, help="Candidates to rerank (default: 25)."
    )
    evaluate.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
    evaluate.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    synth = subparsers.add_parser(
        "evaluate-synthesis",
        help="Score a synthesis provider on a benchmark: abstention, citations, cost.",
    )
    synth.add_argument("benchmark", type=Path, help="Benchmark directory or questions.json.")
    synth.add_argument(
        "--synthesis",
        choices=("extractive", "llm"),
        default="extractive",
        help="Provider to score (default: extractive, which needs no API key).",
    )
    synth.add_argument("--llm-model", default=None, help="Model name, for --synthesis llm.")
    synth.add_argument(
        "--k", type=int, default=8, help="Passages retrieved as evidence (default: 8)."
    )
    synth.add_argument(
        "--method",
        choices=("bm25", "dense", "hybrid"),
        default="bm25",
        help="Retrieval method feeding synthesis (default: bm25).",
    )
    synth.add_argument("--model", default=None, help="Embedding model, for dense and hybrid.")
    synth.add_argument(
        "--rerank", action="store_true", help="Reorder the shortlist before synthesis."
    )
    synth.add_argument("--rerank-model", default=None, help="Cross-encoder model, for --rerank.")
    synth.add_argument(
        "--rerank-pool", type=int, default=25, help="Candidates to rerank (default: 25)."
    )
    synth.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the full JSON report here, including every claim and its cited text.",
    )
    synth.add_argument(
        "--db", type=Path, default=default_db, help=f"SQLite file (default: {default_db})."
    )

    faith = subparsers.add_parser(
        "judge-faithfulness",
        help="Judge whether each claim in a synthesis report follows from its cited passages.",
    )
    faith.add_argument("report", type=Path, help="A report written by evaluate-synthesis --out.")
    faith.add_argument(
        "--worksheet",
        type=Path,
        default=None,
        help="Write a sample for hand labelling and stop. Needs no API key.",
    )
    faith.add_argument(
        "--sample", type=int, default=20, help="Claims to draw for --worksheet (default: 20)."
    )
    faith.add_argument(
        "--seed", type=int, default=1, help="Sampling seed, fixed so a draw is repeatable."
    )
    faith.add_argument("--judge-model", default=None, help="Model used to judge.")
    faith.add_argument("--out", type=Path, default=None, help="Write the full JSON report here.")
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


def _retriever_factory(args: argparse.Namespace, runtime: Runtime) -> RetrieverFactory:
    """The base method, optionally wrapped in a reranking stage."""
    factory = _build_retriever_factory(args, runtime)
    if not getattr(args, "rerank", False):
        return factory
    reranker = runtime.build_reranker(args.rerank_model or runtime.default_rerank_model)
    return with_reranking(factory, reranker, pool=args.rerank_pool)


def _cmd_query(args: argparse.Namespace, runtime: Runtime) -> int:
    build = _retriever_factory(args, runtime)
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
    build = _retriever_factory(args, runtime)
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


def render_synthesis_summary(report: SynthesisReport) -> str:
    """The scoreboard. Citation integrity is absent on purpose: see the module docstring."""
    a = report.abstention
    lines = [
        f"Benchmark {report.benchmark_version}"
        f"  retrieval={report.retriever}  synthesis={report.synthesizer}",
        f"Questions: {report.questions}",
        "",
        "Abstention",
        f"  answerable {a.answerable}: answered {a.answered}, refused {a.false_abstentions}",
        f"  unanswerable {a.unanswerable}: abstained {a.abstained}, answered {a.false_answers}",
    ]
    if a.false_answers:
        lines.append("  WARNING: answered a question the benchmark marks unanswerable")

    precision = report.mean_citation_precision
    lines += [
        "",
        "Citations",
        f"  cited papers judged relevant: {_pct(precision)}",
        f"  claims per answer: {report.mean_claims_per_answer:.1f}"
        f"  papers per answer: {report.mean_papers_per_answer:.1f}",
        f"  fabrication attempts (ids not supplied): {report.unsupported_citations}",
        "",
        "Cost",
        f"  tokens: {report.total_input_tokens} in / {report.total_output_tokens} out",
        f"  mean latency: {report.mean_latency_ms:.0f} ms per question",
        "",
        "By question type",
    ]
    for qtype, summary in report.by_type.items():
        lines.append(
            f"  {qtype:<13} answered {summary.answered}/{summary.questions}"
            f"  claims {summary.mean_claims:.1f}"
            f"  citation precision {_pct(summary.mean_citation_precision)}"
        )
    lines += [
        "",
        "Faithfulness is not scored here; it needs judgment. Use --out and judge the claims.",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _cmd_evaluate_synthesis(args: argparse.Namespace, runtime: Runtime) -> int:
    if not args.benchmark.exists():
        print(f"error: cannot read {args.benchmark}: not found", file=sys.stderr)
        return EXIT_BAD_INPUT
    benchmark = load_benchmark(args.benchmark)
    build = _retriever_factory(args, runtime)
    provider = _synthesizer(args, runtime)
    with runtime.open_repository(args.db) as repository:
        report = evaluate_synthesis(
            repository,
            benchmark,
            build_retriever=build,
            synthesizer=provider,
            evidence_limit=args.k,
        )
    print(render_synthesis_summary(report))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"report written to {args.out}")
    return EXIT_OK


def render_faithfulness_summary(report: FaithfulnessReport) -> str:
    lines = [
        f"Synthesis {report.synthesizer}  judged by {report.judge}",
        f"Claims: {report.claims}",
        "",
        f"  supported            {report.supported}",
        f"  partially supported  {report.partially_supported}",
        f"  unsupported          {report.unsupported}",
        "",
        f"  strict  (supported only)       {report.strict_rate:.3f}",
        f"  lenient (supported or partial) {report.lenient_rate:.3f}",
        "",
        f"  judge tokens: {report.input_tokens} in / {report.output_tokens} out",
        "",
        "Strict rate by question type",
    ]
    lines += [f"  {qtype:<13} {rate:.3f}" for qtype, rate in report.by_type.items()]
    lines += [
        "",
        "A model graded a model. Label a sample by hand and compare before trusting these.",
    ]
    return NEWLINE.join(lines)


def _cmd_judge_faithfulness(args: argparse.Namespace, runtime: Runtime) -> int:
    if not args.report.exists():
        print(f"error: cannot read {args.report}: not found", file=sys.stderr)
        return EXIT_BAD_INPUT
    source = SynthesisReport.model_validate_json(args.report.read_text(encoding="utf-8"))

    if args.worksheet is not None:
        try:
            sheet = build_worksheet(source, size=args.sample, seed=args.seed)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        _write_json(args.worksheet, sheet.model_dump(mode="json"))
        print(
            f"{len(sheet.items)} of {sheet.population} claims written to {args.worksheet}."
            " Fill in each verdict field, then compare."
        )
        return EXIT_OK

    judge = runtime.build_judge(args.judge_model or runtime.default_judge_model)
    try:
        report = judge_report(source, judge.judge, judge.name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    print(render_faithfulness_summary(report))
    if args.out is not None:
        _write_json(args.out, report.model_dump(mode="json"))
        print(f"report written to {args.out}")
    return EXIT_OK


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + NEWLINE, encoding="utf-8")


def _cmd_embed(args: argparse.Namespace, runtime: Runtime) -> int:
    model = runtime.build_embedding_model(args.model or runtime.default_embedding_model)
    with runtime.open_repository(args.db) as repository:
        report = embed_corpus(repository, model, batch_size=args.batch_size)
    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return EXIT_OK


def render_answer(answer: ResearchAnswer, papers: dict[str, Paper]) -> str:
    """The answer, its evidence, and what it does not cover."""
    lines: list[str] = [f"Question: {answer.question}", ""]

    if answer.abstained:
        lines.append("No answer: the retrieved evidence does not support one.")
    else:
        for index, citation in enumerate(answer.citations, start=1):
            refs = ", ".join(citation.evidence_ids)
            lines.append(f"{index}. {citation.claim}")
            lines.append(f"   [{refs}]")
        lines.append("")
        lines.append("Sources")
        for chunk_id in dict.fromkeys(
            ref for citation in answer.citations for ref in citation.evidence_ids
        ):
            passage = next(
                (e.passage for e in answer.evidence if e.passage.chunk_id == chunk_id), None
            )
            if passage is None:
                continue
            paper = papers.get(passage.source_id)
            title = paper.title if paper else "(title unavailable)"
            url = paper.url if paper else ""
            lines.append(f"  {chunk_id}  {title}")
            if url:
                lines.append(f"      {url}")

    lines.append("")
    lines.append("Limitations")
    lines.extend(f"  - {note}" for note in answer.limitations)
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _synthesizer(args: argparse.Namespace, runtime: Runtime) -> SynthesisProvider | None:
    """The chosen provider, or ``None`` to let the service use its default."""
    if args.synthesis != "llm":
        return None
    return runtime.build_synthesizer(args.llm_model or runtime.default_llm_model)


def _report_usage(provider: SynthesisProvider | None) -> None:
    """Cost and latency to stderr, so stdout stays pipeable in either format."""
    usage = getattr(provider, "last_usage", None)
    if usage is None:
        return
    note = (
        f"synthesis: {usage.input_tokens} in / {usage.output_tokens} out tokens"
        f" in {usage.latency_seconds:.2f}s"
    )
    if usage.unsupported_citations:
        note += f", {usage.unsupported_citations} unsupported citation(s) dropped"
    print(note, file=sys.stderr)


def _cmd_answer(args: argparse.Namespace, runtime: Runtime) -> int:
    build = _retriever_factory(args, runtime)
    provider = _synthesizer(args, runtime)
    with runtime.open_repository(args.db) as repository:
        answer = answer_question(
            repository,
            args.question,
            build_retriever=build,
            synthesizer=provider,
            evidence_limit=args.k,
            char_budget=args.char_budget,
            min_score=args.min_score,
        )
        cited = {
            passage.source_id
            for passage in (e.passage for e in answer.evidence)
            if any(passage.chunk_id in c.evidence_ids for c in answer.citations)
        }
        papers = {
            pid: paper
            for pid in cited
            if (paper := repository.get_paper(Source.PUBMED, pid)) is not None
        }

    if args.format == "json":
        print(json.dumps(answer.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        print(render_answer(answer, papers))
    _report_usage(provider)
    return EXIT_OK


_COMMANDS: dict[str, Callable[[argparse.Namespace, Runtime], int]] = {
    "search": _cmd_search,
    "ingest": _cmd_ingest,
    "show": _cmd_show,
    "passages": _cmd_passages,
    "query": _cmd_query,
    "answer": _cmd_answer,
    "load-corpus": _cmd_load_corpus,
    "judge": _cmd_judge,
    "embed": _cmd_embed,
    "evaluate": _cmd_evaluate,
    "evaluate-synthesis": _cmd_evaluate_synthesis,
    "judge-faithfulness": _cmd_judge_faithfulness,
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
    except (
        LiteratureSourceError,
        PersistenceError,
        EmbeddingError,
        RerankerError,
        SynthesisError,
    ) as exc:
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
            build_embedding_model=lambda name: build_embedding_model(
                name, cache_dir=settings.embedding_cache_dir
            ),
            default_embedding_model=settings.embedding_model,
            build_reranker=lambda name: build_reranker(
                name, cache_dir=settings.embedding_cache_dir
            ),
            default_rerank_model=settings.rerank_model,
            build_synthesizer=lambda name: LlmSynthesizer(
                api_key=settings.anthropic_api_key,
                model=name,
                max_tokens=settings.llm_max_tokens,
            ),
            default_llm_model=settings.llm_model,
            build_judge=lambda name: LlmFaithfulnessJudge(
                api_key=settings.anthropic_api_key, model=name
            ),
            default_judge_model=settings.judge_model,
        )
        return run(sys.argv[1:] if argv is None else argv, runtime)


if __name__ == "__main__":
    sys.exit(main())
