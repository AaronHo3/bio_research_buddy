"""Turning retrieved evidence into a cited answer."""

from bio_research_copilot.synthesis.base import SynthesisError, SynthesisProvider
from bio_research_copilot.synthesis.context import DEFAULT_CHAR_BUDGET, select_evidence
from bio_research_copilot.synthesis.extractive import ExtractiveSynthesizer
from bio_research_copilot.synthesis.llm import (
    DEFAULT_MAX_TOKENS,
    LlmSynthesizer,
    Usage,
    build_llm_synthesizer,
)
from bio_research_copilot.synthesis.llm import DEFAULT_MODEL as DEFAULT_LLM_MODEL

__all__ = [
    "DEFAULT_CHAR_BUDGET",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_MAX_TOKENS",
    "ExtractiveSynthesizer",
    "LlmSynthesizer",
    "SynthesisError",
    "SynthesisProvider",
    "Usage",
    "build_llm_synthesizer",
    "select_evidence",
]
