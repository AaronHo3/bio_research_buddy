"""Fitting evidence into a budget before synthesis.

Every synthesis provider works against a limit: a language model has a context
window, and even the extractive baseline should not be handed unbounded text.
Selecting evidence is therefore part of the pipeline rather than a detail of any
one provider, and it happens in rank order so that what gets dropped is always
the worst-ranked material.

The budget is counted in characters. Characters are not tokens, and the ratio
differs by model and by text, so a provider with a hard token limit must apply
its own check as well. Characters are used here because they are exact, cost
nothing to compute, and do not tie the pipeline to one model's tokeniser.
"""

from bio_research_copilot.domain import RetrievedEvidence

DEFAULT_CHAR_BUDGET = 8000


def select_evidence(
    evidence: list[RetrievedEvidence], *, char_budget: int = DEFAULT_CHAR_BUDGET
) -> list[RetrievedEvidence]:
    """Highest-ranked evidence that fits the budget, in rank order.

    Always returns at least one item when given any, even if that item alone
    exceeds the budget; handing a provider nothing when evidence exists would
    turn a long passage into a silent abstention.
    """
    if char_budget < 1:
        msg = "char_budget must be at least 1"
        raise ValueError(msg)

    ordered = sorted(evidence, key=lambda item: item.rank)
    selected: list[RetrievedEvidence] = []
    used = 0
    for item in ordered:
        cost = len(item.passage.text)
        if selected and used + cost > char_budget:
            break
        selected.append(item)
        used += cost
    return selected
