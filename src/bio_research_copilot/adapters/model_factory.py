"""Choose the right adapter for a model name.

The composition root asks for a model by name and gets back something that
satisfies the port, without knowing which backend serves it. MedCPT is imported
lazily so that PyTorch, an optional dependency, is only needed by someone who
actually asks for MedCPT.
"""

from pathlib import Path

from bio_research_copilot.adapters.embedding import EmbeddingModel
from bio_research_copilot.adapters.fastembed_model import FastEmbedModel
from bio_research_copilot.adapters.fastembed_reranker import FastEmbedReranker
from bio_research_copilot.adapters.reranking import Reranker

MEDCPT_PREFIX = "ncbi/MedCPT"


def build_embedding_model(name: str, cache_dir: Path | None = None) -> EmbeddingModel:
    """An embedding model for ``name``. Local ONNX by default, MedCPT on request."""
    if name.startswith(MEDCPT_PREFIX):
        from bio_research_copilot.adapters.medcpt import MedCptEmbeddingModel

        return MedCptEmbeddingModel(cache_dir=cache_dir)
    return FastEmbedModel(name, cache_dir=cache_dir)


def build_reranker(name: str, cache_dir: Path | None = None) -> Reranker:
    """A reranker for ``name``. Local ONNX cross-encoder by default, MedCPT on request."""
    if name.startswith(MEDCPT_PREFIX):
        from bio_research_copilot.adapters.medcpt import MedCptReranker

        return MedCptReranker(cache_dir=cache_dir)
    return FastEmbedReranker(name, cache_dir=cache_dir)
