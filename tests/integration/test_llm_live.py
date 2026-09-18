"""Live checks against the real Anthropic API.

Deselected by default and skipped without a key, because these cost money and
need a network. They exist to catch the things a mock cannot: that the tool
schema is accepted by the real API, and that a real model actually honours the
instruction to cite only supplied ids.
"""

import hashlib
import os

import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, Source
from bio_research_copilot.synthesis.llm import LlmSynthesizer

pytestmark = pytest.mark.live

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
requires_key = pytest.mark.skipif(not API_KEY, reason="ANTHROPIC_API_KEY is not set")

QUESTION = "What causes off-target editing by base editors?"


def evidence(chunk: str, text: str, rank: int) -> RetrievedEvidence:
    pmid, position = chunk.split(":")
    return RetrievedEvidence(
        passage=Passage(
            source=Source.PUBMED,
            source_id=pmid,
            chunk_id=f"pubmed:{chunk}",
            position=int(position),
            section="abstract",
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            strategy="section-v1",
        ),
        rank=rank,
        score=1.0 / rank,
        method="bm25-v1",
    )


EVIDENCE = [
    evidence(
        "31978945:0",
        "Cytosine base editors produced off-target edits arising from deaminase"
        " activity on transiently exposed single-stranded DNA, independent of the"
        " guide RNA.",
        1,
    ),
    evidence(
        "29160308:0",
        "Guide RNA design influenced editing specificity, with extended guides"
        " reducing bystander edits within the activity window.",
        2,
    ),
]


@requires_key
def test_a_real_model_returns_claims_citing_only_supplied_ids() -> None:
    provider = LlmSynthesizer(api_key=API_KEY)
    citations = provider.synthesize(QUESTION, EVIDENCE)

    assert citations, "the evidence answers the question, so claims were expected"
    available = {item.passage.chunk_id for item in EVIDENCE}
    for citation in citations:
        assert set(citation.evidence_ids) <= available

    usage = provider.last_usage
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.unsupported_citations == 0, "the model invented an evidence id"


@requires_key
def test_a_real_model_abstains_when_the_evidence_is_unrelated() -> None:
    """The rule that matters most: no answer is better than a wrong one."""
    unrelated = [
        evidence("40000001:0", "Zebrafish were housed at twenty-eight degrees Celsius.", 1),
        evidence("40000002:0", "Tank water was exchanged twice weekly.", 2),
    ]
    provider = LlmSynthesizer(api_key=API_KEY)
    assert provider.synthesize(QUESTION, unrelated) == []
