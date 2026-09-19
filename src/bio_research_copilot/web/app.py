"""The public demo: an HTTP surface over the same application services as the CLI.

A visitor's question is answered against PubMed itself, live, so the reachable
corpus is the whole database rather than a frozen sample. Nothing is stored
between requests: papers are fetched, ranked in memory, and discarded.

Synthesis is by a language model while the day's token budget lasts, and by the
deterministic extractive provider once it does not. The page always states which
one answered, because the evaluation measures a real difference between them and
quietly substituting one for the other would misrepresent the system. Running out
of budget is therefore a visible degradation rather than an outage.

The example questions are separate and static: answers recorded during a measured
evaluation run, replayed from a file. They cost nothing to serve, so the page has
something good to show even with the budget spent or no key configured at all.

Three guards stand between this and an unbounded bill, described in ``guards``.
Together they mean the worst case for a day is the configured number of tokens.
"""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from bio_research_copilot.adapters.literature import LiteratureSource, LiteratureSourceError
from bio_research_copilot.adapters.query_planner import QueryPlanner
from bio_research_copilot.application.demo_export import DemoClaim, DemoData, DemoSource
from bio_research_copilot.application.live_answer import FullTextSource, answer_live
from bio_research_copilot.synthesis import SynthesisError, SynthesisProvider
from bio_research_copilot.web.guards import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_DAILY_TOKENS,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW,
    AnswerCache,
    RateLimiter,
    TokenBudget,
)

STATIC = Path(__file__).parent / "static"
DEFAULT_DATA = Path("web/data/answers.json")

MAX_QUESTION_CHARS = 300
EVIDENCE_CHAR_BUDGET = 14000
"""Full-text chunks are far larger than abstracts, so the default budget would
admit only a handful. Raised to keep roughly eight passages in play."""
PAPERS_PER_QUESTION = 30
EVIDENCE_LIMIT = 8

EXTRACTIVE_NOTE = (
    "Answered by the deterministic extractive baseline, which copies sentences"
    " verbatim and cannot combine or rephrase them."
)
BUDGET_NOTE = "The daily language-model budget is spent, so this used the extractive baseline."


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AnswerResponse(BaseModel):
    """An answer shaped like a stored one, so the page renders both the same."""

    question: str
    synthesizer: str
    retriever: str
    model_backed: bool = Field(description="Whether a language model wrote the claims.")
    cached: bool
    abstained: bool
    abstain_reason: str | None
    claims: list[DemoClaim]
    sources: list[DemoSource]
    passages: dict[str, str] = Field(
        default_factory=dict,
        description="Cited passage text by chunk id, so a claim can be checked in place.",
    )
    limitations: list[str]
    papers_fetched: int
    full_text_papers: int = Field(
        default=0, description="Papers read in full rather than by abstract."
    )
    search_term: str | None = Field(default=None, description="The PubMed query actually searched.")


def load_demo_data(path: Path) -> DemoData:
    """The pre-computed example answers.

    Raises:
        FileNotFoundError: the export has not been generated yet.
    """
    if not path.exists():
        msg = (
            f"{path} is missing; generate it with"
            " 'bio-copilot export-demo <report.json> --out web/data/answers.json'"
        )
        raise FileNotFoundError(msg)
    return DemoData.model_validate_json(path.read_text(encoding="utf-8"))


def create_app(
    *,
    source: LiteratureSource,
    full_text_source: FullTextSource | None = None,
    data_path: Path = DEFAULT_DATA,
    build_synthesizer: Callable[[], SynthesisProvider] | None = None,
    build_planner: Callable[[], QueryPlanner] | None = None,
    daily_tokens: int = DEFAULT_DAILY_TOKENS,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window: float = DEFAULT_RATE_WINDOW,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> FastAPI:
    """Build the application.

    Args:
        source: The literature source. Injected, so tests never touch NCBI.
        full_text_source: Supplies bodies from PubMed Central. ``None`` answers
            from abstracts alone, which is a degradation, not a failure.
        data_path: The exported example answers.
        build_synthesizer: Makes a language-model provider. ``None`` means no
            key is configured and every answer is extractive.
        build_planner: Makes a query planner. ``None`` leaves the source to
            translate the question itself, which is worse but free.
        daily_tokens: Ceiling on tokens spent per day. Zero disables the model.
    """
    demo = load_demo_data(data_path)
    budget = TokenBudget(daily_tokens)
    cache = AnswerCache(max_entries=cache_size)
    limiter = RateLimiter(rate_limit, rate_window)

    app = FastAPI(
        title="Bio Research Copilot",
        description="Evidence-grounded answers over live PubMed literature.",
        version="0.2.0",
    )

    def synthesizer() -> tuple[SynthesisProvider | None, bool]:
        """The provider to use now, and whether it is model-backed."""
        if build_synthesizer is None or not budget.allows():
            return None, False
        try:
            return build_synthesizer(), True
        except SynthesisError:
            # A missing key or package must degrade, not take the demo down.
            return None, False

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        status = budget.status()
        return {
            "status": "ok",
            "examples": len(demo.answers),
            "model_available": build_synthesizer is not None and not status.exhausted,
            "budget": {
                "limit": status.limit,
                "spent": status.spent,
                "remaining": status.remaining,
            },
            "cache": {"entries": len(cache), "hits": cache.hits, "misses": cache.misses},
        }

    @app.get("/api/examples", response_model=DemoData)
    def examples() -> DemoData:
        """Recorded model answers. Static data; no provider is called."""
        return demo

    @app.post("/api/answer", response_model=AnswerResponse)
    def ask(request: AskRequest, http: Request) -> AnswerResponse:
        """Answer a question against live PubMed."""
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question cannot be blank")

        client = http.client.host if http.client else "unknown"
        if not limiter.allows(client):
            raise HTTPException(
                status_code=429,
                detail="Too many questions from this address. Try again in a minute.",
            )

        hit = cache.get(question)
        if isinstance(hit, AnswerResponse):
            return hit.model_copy(update={"cached": True})

        provider, model_backed = synthesizer()
        # Planning shares the synthesis budget: both spend the same money, and a
        # ceiling that ignored one of them would not be a ceiling.
        planner = build_planner() if build_planner is not None and budget.allows() else None
        try:
            result = answer_live(
                source,
                question,
                full_text_source=full_text_source,
                planner=planner,
                synthesizer=provider,
                papers=PAPERS_PER_QUESTION,
                evidence_limit=EVIDENCE_LIMIT,
                char_budget=EVIDENCE_CHAR_BUDGET,
            )
        except LiteratureSourceError as exc:
            raise HTTPException(
                status_code=503, detail=f"PubMed could not be reached: {exc}"
            ) from exc
        except SynthesisError as exc:
            raise HTTPException(status_code=502, detail=f"Synthesis failed: {exc}") from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        # Whether a model actually wrote anything, not merely whether one was
        # offered: retrieval can abstain before synthesis is ever reached, and
        # reporting that answer as model-backed would misdescribe it.
        planning = getattr(planner, "last_usage", None) if planner is not None else None
        if planning is not None:
            budget.charge(planning.input_tokens + planning.output_tokens)
        usage = getattr(provider, "last_usage", None) if provider is not None else None
        model_used = model_backed and usage is not None
        if usage is not None:
            budget.charge(usage.input_tokens + usage.output_tokens)

        response = _to_response(result, model_backed=model_used, degraded=_degraded(model_backed))
        cache.put(question, response)
        return response

    def _degraded(model_backed: bool) -> str | None:
        if model_backed:
            return None
        if build_synthesizer is None:
            return EXTRACTIVE_NOTE
        return BUDGET_NOTE

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def _to_response(result: Any, *, model_backed: bool, degraded: str | None) -> AnswerResponse:
    answer = result.answer
    by_chunk = {item.passage.chunk_id: item.passage.source_id for item in answer.evidence}
    claims = [
        DemoClaim(
            text=citation.claim,
            chunk_ids=list(citation.evidence_ids),
            paper_ids=_unique(by_chunk[ref] for ref in citation.evidence_ids if ref in by_chunk),
        )
        for citation in answer.citations
    ]
    cited = {pid for claim in claims for pid in claim.paper_ids}
    passages = {
        item.passage.chunk_id: item.passage.text
        for item in answer.evidence
        if any(item.passage.chunk_id in c.evidence_ids for c in answer.citations)
    }
    sources = [
        DemoSource(
            paper_id=paper.source_id,
            title=paper.title,
            url=paper.url,
            journal=paper.journal,
            year=paper.publication_date.year if paper.publication_date else None,
        )
        for pid, paper in result.papers.items()
        if pid in cited
    ]
    limitations = list(answer.limitations)
    if degraded:
        limitations.append(degraded)

    return AnswerResponse(
        question=answer.question,
        synthesizer=answer.synthesis_method,
        retriever=answer.retrieval_method,
        model_backed=model_backed,
        cached=False,
        abstained=answer.abstained,
        abstain_reason=answer.limitations[0] if answer.abstained else None,
        claims=claims,
        sources=sources,
        passages=passages,
        limitations=limitations,
        papers_fetched=result.papers_fetched,
        full_text_papers=result.full_text_papers,
        search_term=result.plan.term if result.plan else None,
    )


def _unique(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def build_default_app() -> FastAPI:
    """The application wired from settings. Used by the server entry point."""
    import httpx

    from bio_research_copilot.adapters.pubmed import PubMedClient
    from bio_research_copilot.adapters.query_planner import LlmQueryPlanner
    from bio_research_copilot.config import Settings
    from bio_research_copilot.synthesis.llm import LlmSynthesizer

    settings = Settings()
    http = httpx.Client(timeout=settings.http_timeout_seconds)
    source = PubMedClient(
        http,
        tool=settings.ncbi_tool_name,
        email=settings.ncbi_contact_email,
        api_key=settings.ncbi_api_key,
    )

    key = settings.anthropic_api_key

    def make_synthesizer() -> SynthesisProvider:
        return LlmSynthesizer(
            api_key=key, model=settings.llm_model, max_tokens=settings.llm_max_tokens
        )

    def make_planner() -> QueryPlanner:
        return LlmQueryPlanner(api_key=key, model=settings.planner_model)

    return create_app(
        source=source,
        full_text_source=source,
        build_synthesizer=make_synthesizer if key else None,
        build_planner=make_planner if key else None,
        daily_tokens=settings.demo_daily_tokens,
    )


def main() -> None:
    """Run the demo locally: ``uv run bio-copilot-web``."""
    import uvicorn

    uvicorn.run(build_default_app(), host="127.0.0.1", port=8000)
