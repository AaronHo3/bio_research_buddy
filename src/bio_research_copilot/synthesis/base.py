"""The port every synthesis provider must implement.

A provider turns a question and a set of evidence into claims, each carrying the
ids of the evidence it rests on. It never sees the store, never retrieves, and
cannot cite anything outside what it was handed. That is what makes grounding
checkable rather than hoped for.
"""

from typing import Protocol

from bio_research_copilot.domain import Citation, RetrievedEvidence


class SynthesisError(Exception):
    """The provider could not produce a usable answer."""


class SynthesisProvider(Protocol):
    name: str
    """Provider name and version, recorded on every answer."""

    def synthesize(self, question: str, evidence: list[RetrievedEvidence]) -> list[Citation]:
        """Claims supporting an answer, best first, each citing supplied evidence only.

        Returning an empty list is a valid outcome and means the evidence did
        not support a claim. The caller turns that into an abstention.

        Raises:
            SynthesisError: the provider failed to produce output at all.
        """
        ...
