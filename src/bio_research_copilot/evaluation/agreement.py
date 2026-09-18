"""Agreement between two sets of labels over the same items.

Raw agreement is the fraction of items labelled identically. It is easy to read
and easy to be misled by: if nearly every item belongs to one category, two
labellers who both guess that category agree almost always while demonstrating
nothing.

Cohen's kappa corrects for that by subtracting the agreement expected if both
labellers assigned their own category frequencies at random. It has its own
well-known pathology in the other direction: when the marginals are heavily
skewed, expected agreement approaches observed agreement and kappa collapses
toward zero even though the labellers almost never differ. Neither number is
sufficient alone, so both are reported together with the confusion matrix that
explains them.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field


class Disagreement(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    left: str
    right: str


class AgreementReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: int
    agreed: int
    raw_agreement: float
    expected_agreement: float
    kappa: float | None = Field(
        description="None when every label is identical, which makes kappa undefined."
    )
    left_counts: dict[str, int]
    right_counts: dict[str, int]
    confusion: dict[str, dict[str, int]] = Field(description="confusion[left][right] = count.")
    disagreements: list[Disagreement]


def compare_labels(
    left: Mapping[str, str],
    right: Mapping[str, str],
    *,
    categories: Sequence[str],
) -> AgreementReport:
    """Compare two labellings keyed by item id.

    Only keys present in both are compared, because an item one side did not
    label is missing data rather than a disagreement.

    Args:
        left: The first labelling, conventionally the human's.
        right: The second labelling, conventionally the machine's.
        categories: Every label that could have been assigned. Passed in rather
            than inferred, so a category nobody used still appears in the
            confusion matrix instead of silently vanishing.

    Raises:
        ValueError: no keys are shared, or a label is outside ``categories``.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        msg = "the two labellings share no items"
        raise ValueError(msg)

    allowed = set(categories)
    for key in shared:
        for label in (left[key], right[key]):
            if label not in allowed:
                msg = f"label {label!r} for {key} is not one of {sorted(allowed)}"
                raise ValueError(msg)

    total = len(shared)
    agreed = sum(1 for key in shared if left[key] == right[key])
    left_counts = {c: sum(1 for k in shared if left[k] == c) for c in categories}
    right_counts = {c: sum(1 for k in shared if right[k] == c) for c in categories}

    expected = sum((left_counts[c] / total) * (right_counts[c] / total) for c in categories)
    observed = agreed / total
    # Perfect expected agreement means one category was used exclusively by both
    # labellers; there is no chance-corrected signal to extract.
    kappa = None if expected >= 1.0 else (observed - expected) / (1.0 - expected)

    confusion = {
        a: {b: sum(1 for k in shared if left[k] == a and right[k] == b) for b in categories}
        for a in categories
    }

    return AgreementReport(
        items=total,
        agreed=agreed,
        raw_agreement=observed,
        expected_agreement=expected,
        kappa=kappa,
        left_counts=left_counts,
        right_counts=right_counts,
        confusion=confusion,
        disagreements=[
            Disagreement(key=key, left=left[key], right=right[key])
            for key in shared
            if left[key] != right[key]
        ],
    )


def interpret_kappa(kappa: float | None) -> str:
    """The conventional Landis and Koch reading, which is a convention not a law."""
    if kappa is None:
        return "undefined (one category used exclusively)"
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"
