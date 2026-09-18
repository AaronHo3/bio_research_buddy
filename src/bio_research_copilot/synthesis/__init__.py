"""Turning retrieved evidence into a cited answer."""

from bio_research_copilot.synthesis.base import SynthesisError, SynthesisProvider
from bio_research_copilot.synthesis.context import DEFAULT_CHAR_BUDGET, select_evidence
from bio_research_copilot.synthesis.extractive import ExtractiveSynthesizer

__all__ = [
    "DEFAULT_CHAR_BUDGET",
    "ExtractiveSynthesizer",
    "SynthesisError",
    "SynthesisProvider",
    "select_evidence",
]
