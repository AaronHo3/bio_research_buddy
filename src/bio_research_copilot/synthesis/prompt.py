"""The text and schema a language model is given, kept apart from the call.

Prompt construction is a pure function of question and evidence, so it can be
asserted against a golden file. Wording changes then appear as a reviewable diff
rather than as an unexplained shift in answer quality.

Two details carry the grounding design. Passages are labelled with their real
``chunk_id`` rather than with positions, so a citation is either a genuine id or
an obvious error, with no index-to-id translation step in between. And the claim
schema makes ``evidence_ids`` a required, non-empty array, so a claim without
support is not expressible in the model's output at all.
"""

from typing import Any

from bio_research_copilot.domain import RetrievedEvidence

TOOL_NAME = "record_claims"

SYSTEM_PROMPT = """\
You are a biomedical literature assistant. You answer strictly from the evidence \
passages supplied to you in each request.

Rules:

1. Every claim must be supported by the supplied passages and must cite them by \
their exact id.
2. Use only the supplied passages. Do not add facts from your own knowledge, \
even where you are confident they are correct.
3. A claim may cite several passages when they jointly support it. Prefer that \
over repeating a claim, because a corroborated statement is more useful than two \
separate ones.
4. Write each claim as a complete, standalone sentence that reads sensibly \
without the question beside it. Do not write a bare title or a sentence fragment.
5. Where passages disagree, say that they disagree rather than choosing a side.
6. Do not give medical advice or make recommendations about patient care.
7. If the passages do not support an answer to the question, record no claims at \
all. That is a valid and useful outcome: no answer is better than a wrong one.\
"""

CLAIM_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record the claims that answer the user's question, each mapped to the"
        " evidence passages supporting it. Record an empty list when the"
        " evidence does not support an answer."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "description": "Claims in order of importance, most important first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": (
                                "A complete standalone sentence supported by the cited passages."
                            ),
                        },
                        "evidence_ids": {
                            "type": "array",
                            "description": (
                                "Ids of supplied passages supporting this claim, copied exactly."
                            ),
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["claim", "evidence_ids"],
                },
            }
        },
        "required": ["claims"],
    },
}


def format_evidence(evidence: list[RetrievedEvidence]) -> str:
    """The evidence block, one tagged passage per item, in the order given."""
    blocks = [
        f'<passage id="{item.passage.chunk_id}" section="{item.passage.section}">\n'
        f"{item.passage.text}\n"
        f"</passage>"
        for item in evidence
    ]
    return "\n".join(blocks)


def build_user_prompt(question: str, evidence: list[RetrievedEvidence], *, max_claims: int) -> str:
    """The full request: evidence first, then the question.

    Evidence precedes the question so the model reads the material before it
    knows what to look for, which discourages it from hunting for confirmation
    of an answer it has already formed.
    """
    return (
        "<evidence>\n"
        f"{format_evidence(evidence)}\n"
        "</evidence>\n\n"
        f"Question: {question}\n\n"
        f"Record at most {max_claims} claims that answer this question, using only"
        " the passages above. Cite each passage by the exact id in its tag."
    )
