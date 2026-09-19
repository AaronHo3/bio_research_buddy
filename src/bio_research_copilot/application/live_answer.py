"""Answer a question against PubMed itself, with nothing stored in between.

The stored pipeline ingests a corpus once and ranks it many times. This one
inverts that: it searches PubMed for the question, fetches a few dozen papers,
chunks and ranks them in memory, synthesises, and throws the index away.

That sounds wasteful and is not. BM25 over thirty papers is a few hundred
passages, which builds in milliseconds, so the index is far cheaper than the
network call that produced its input. In exchange the whole of PubMed becomes
reachable without a vector database, without an ingestion job, and without any
index that can drift out of date with the source.

What it costs is honesty about freshness and failure. The answer depends on
whichever papers PubMed returned for this phrasing at this moment, so the same
question tomorrow may retrieve different evidence. And NCBI being unreachable is
now a user-visible failure rather than a background job retrying quietly, which
is why the source error is allowed to propagate rather than being swallowed into
an empty answer that would look like a considered abstention.
"""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from bio_research_copilot.adapters.literature import LiteratureSource
from bio_research_copilot.adapters.pmc import FullText
from bio_research_copilot.adapters.query_planner import QueryPlanner, SearchPlan
from bio_research_copilot.domain import (
    Paper,
    Passage,
    ResearchAnswer,
    RetrievedEvidence,
    SearchQuery,
)
from bio_research_copilot.domain.chunking import FullTextChunker
from bio_research_copilot.retrieval.bm25 import BM25Index
from bio_research_copilot.synthesis import (
    DEFAULT_CHAR_BUDGET,
    ExtractiveSynthesizer,
    SynthesisProvider,
    select_evidence,
)

DEFAULT_PAPERS = 30
"""Papers to fetch per question. Enough for BM25 to discriminate, few enough to
stay inside one efetch call and a few seconds."""

DEFAULT_EVIDENCE_LIMIT = 8

DEFAULT_FULL_TEXT_PAPERS = 8
"""Papers whose full text is fetched.

Retrieval happens twice. Abstracts for all thirty papers are ranked first, and
only the best few have their bodies fetched, because a body is tens of thousands
of characters and fetching thirty of them would cost seconds of latency and
megabytes of transfer to rank text that was never going to be cited.
"""

FULL_TEXT_LIMITATION = (
    "Full text was read for {have} of {ranked} top-ranked papers; the rest are"
    " represented by their abstracts, because PubMed Central does not offer"
    " downloadable full text for every paper."
)

ABSTAIN_NO_PAPERS = "PubMed returned no papers for this question."
ABSTAIN_NO_PASSAGES = "The papers PubMed returned had no usable text."
ABSTAIN_NO_CLAIM = "Retrieved passages did not contain a sentence that addresses the question."

ABSTRACT_ONLY_LIMITATION = "No full text was available, so evidence is titles and abstracts only."
FRESHNESS_LIMITATION = (
    "Papers were retrieved from PubMed for this question just now, so the same"
    " question may draw on different evidence later."
)


class FullTextSource(Protocol):
    """Anything that can supply bodies for some papers and stay quiet about the rest."""

    def fetch_full_text(self, pmids: list[str], limit: int = 8) -> dict[str, FullText]: ...


class LiveAnswer(BaseModel):
    """A grounded answer plus the papers it was drawn from.

    The papers travel with the answer because nothing was stored: there is no
    repository a caller could look them up in afterwards.
    """

    model_config = ConfigDict(frozen=True)

    answer: ResearchAnswer
    papers: dict[str, Paper] = Field(description="Cited papers by source id.")
    papers_fetched: int
    passages_indexed: int
    plan: SearchPlan | None = Field(
        default=None, description="The planned query, when one was used."
    )
    full_text_papers: int = Field(
        default=0, description="Papers whose body was read rather than just the abstract."
    )


def answer_live(
    source: LiteratureSource,
    question: str,
    *,
    full_text_source: FullTextSource | None = None,
    planner: QueryPlanner | None = None,
    synthesizer: SynthesisProvider | None = None,
    papers: int = DEFAULT_PAPERS,
    full_text_papers: int = DEFAULT_FULL_TEXT_PAPERS,
    evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> LiveAnswer:
    """Search, read, rank, and answer, holding the index only for this call.

    Retrieval runs twice. Abstracts for every result are ranked to decide which
    papers are worth reading; the bodies of the best few are then fetched and
    chunked, and the chunks are ranked to choose the evidence. Ranking the
    abstracts first is what keeps the expensive step proportional to relevance
    rather than to the size of the result set.

    Raises:
        pydantic.ValidationError: the question is blank.
        LiteratureSourceError: PubMed could not be reached or its reply could
            not be parsed. Deliberately not caught: an infrastructure failure
            must not be presented to a reader as a considered abstention.
        SynthesisError: the synthesis provider failed outright.
    """
    provider = synthesizer or ExtractiveSynthesizer()
    chunker = FullTextChunker()

    query = SearchQuery(raw=question, max_results=papers)
    # A question is not a query. Planning turns one into the other, supplying
    # the vocabulary the literature uses; the source falls back to its own
    # translation when no plan is available.
    plan = planner.plan(query.raw) if planner is not None else None
    found = source.search(query, plan.term if plan else None)
    by_id = {paper.source_id: paper for paper in found}

    def abstain(reason: str, indexed: int = 0, full: int = 0) -> LiveAnswer:
        return LiveAnswer(
            answer=ResearchAnswer(
                question=query.raw,
                citations=[],
                evidence=[],
                limitations=[reason, FRESHNESS_LIMITATION],
                abstained=True,
                retrieval_method=BM25Index.name,
                synthesis_method=provider.name,
            ),
            papers={},
            papers_fetched=len(found),
            passages_indexed=indexed,
            full_text_papers=full,
            plan=plan,
        )

    if not found:
        return abstain(ABSTAIN_NO_PAPERS)

    # First pass: rank abstracts to decide which papers are worth reading.
    abstract_passages = [p for paper in found for p in chunker.split(paper)]
    if not abstract_passages:
        return abstain(ABSTAIN_NO_PASSAGES)
    # Every paper, in relevance order: availability is cheap to check and the
    # adapter spends its one expensive call on the best papers that have a body.
    ranked = _rank_papers(abstract_passages, query.raw, limit=len(found))
    shortlist = ranked[:full_text_papers]

    bodies: dict[str, FullText] = {}
    if full_text_source is not None and ranked:
        bodies = full_text_source.fetch_full_text(ranked, limit=full_text_papers)

    # Second pass: rank chunks, full text where it exists and abstracts elsewhere.
    passages: list[Passage] = []
    for paper in found:
        body = bodies.get(paper.source_id)
        if body is None:
            passages.extend(chunker.split(paper))
        else:
            passages.extend(
                chunker.split_full_text(
                    paper, [(section.label, section.text) for section in body.sections]
                )
            )
    if not passages:
        return abstain(ABSTAIN_NO_PASSAGES, full=len(bodies))

    index = BM25Index(passages)
    evidence = index.search(SearchQuery(raw=query.raw, max_results=evidence_limit))
    if not evidence:
        return abstain(ABSTAIN_NO_PASSAGES, indexed=len(passages), full=len(bodies))

    selected = select_evidence(evidence, char_budget=char_budget)
    citations = provider.synthesize(query.raw, selected)
    if not citations:
        return abstain(ABSTAIN_NO_CLAIM, indexed=len(passages), full=len(bodies))

    answer = ResearchAnswer(
        question=query.raw,
        citations=citations,
        evidence=selected,
        limitations=_limitations(selected, len(found), len(bodies), len(shortlist)),
        abstained=False,
        retrieval_method=index.name,
        synthesis_method=provider.name,
    )
    cited = {
        item.passage.source_id
        for item in selected
        if any(item.passage.chunk_id in c.evidence_ids for c in citations)
    }
    return LiveAnswer(
        answer=answer,
        papers={pid: by_id[pid] for pid in cited if pid in by_id},
        papers_fetched=len(found),
        passages_indexed=len(passages),
        full_text_papers=len(bodies),
        plan=plan,
    )


def _rank_papers(passages: list[Passage], question: str, *, limit: int) -> list[str]:
    """Source ids of the best papers by their abstracts, best first.

    More passages than papers are requested because several can belong to one
    paper, and a shortlist of eight papers cannot be filled from eight passages.
    """
    if limit < 1:
        return []
    index = BM25Index(passages)
    evidence = index.search(SearchQuery(raw=question, max_results=min(100, limit * 4)))
    ordered: list[str] = []
    for item in evidence:
        if item.passage.source_id not in ordered:
            ordered.append(item.passage.source_id)
        if len(ordered) == limit:
            break
    return ordered


def _limitations(
    selected: list[RetrievedEvidence], fetched: int, full_text: int, ranked: int
) -> list[str]:
    papers = {item.passage.source_id for item in selected}
    notes = [
        f"Based on {len(selected)} passages from {len(papers)} papers,"
        f" ranked out of {fetched} PubMed returned.",
    ]
    if full_text:
        notes.append(FULL_TEXT_LIMITATION.format(have=full_text, ranked=ranked))
    else:
        notes.append(ABSTRACT_ONLY_LIMITATION)
    notes.append(FRESHNESS_LIMITATION)
    if len(papers) == 1:
        notes.append("All evidence comes from a single paper, so it is not corroborated.")
    return notes
