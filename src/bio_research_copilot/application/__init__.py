"""Application services: use cases expressed against ports, never against providers."""

from bio_research_copilot.application.embed import EmbedReport, embed_corpus
from bio_research_copilot.application.evaluate import EvaluationReport, evaluate_retrieval
from bio_research_copilot.application.ingest import IngestReport, ingest_papers, load_corpus
from bio_research_copilot.application.query import query_passages
from bio_research_copilot.application.search import search_papers

__all__ = [
    "EmbedReport",
    "EvaluationReport",
    "IngestReport",
    "embed_corpus",
    "evaluate_retrieval",
    "ingest_papers",
    "load_corpus",
    "query_passages",
    "search_papers",
]
