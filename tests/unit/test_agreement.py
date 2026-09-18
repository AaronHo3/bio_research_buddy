"""Tests for label agreement.

Kappa's edge cases are the reason this module exists as its own unit. Perfect
agreement on a single category is undefined rather than 1.0, and heavily skewed
marginals drive kappa toward zero while raw agreement stays high. Both are
counter-intuitive enough that a reader of the evaluation deserves to know they
were handled deliberately.
"""

import pytest

from bio_research_copilot.evaluation import compare_labels, interpret_kappa

CATEGORIES = ("supported", "partially_supported", "unsupported")


class TestAgreement:
    def test_counts_matches(self) -> None:
        report = compare_labels(
            {"a": "supported", "b": "unsupported", "c": "supported"},
            {"a": "supported", "b": "unsupported", "c": "partially_supported"},
            categories=CATEGORIES,
        )
        assert report.items == 3
        assert report.agreed == 2
        assert report.raw_agreement == pytest.approx(2 / 3)

    def test_lists_disagreements_with_both_labels(self) -> None:
        report = compare_labels({"a": "supported"}, {"a": "unsupported"}, categories=CATEGORIES)
        assert len(report.disagreements) == 1
        assert report.disagreements[0].left == "supported"
        assert report.disagreements[0].right == "unsupported"

    def test_only_shared_keys_are_compared(self) -> None:
        """An item one side never labelled is missing data, not a disagreement."""
        report = compare_labels(
            {"a": "supported", "b": "supported"},
            {"a": "supported"},
            categories=CATEGORIES,
        )
        assert report.items == 1

    def test_no_shared_keys_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="share no items"):
            compare_labels({"a": "supported"}, {"b": "supported"}, categories=CATEGORIES)

    def test_a_label_outside_the_categories_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="not one of"):
            compare_labels({"a": "maybe"}, {"a": "supported"}, categories=CATEGORIES)

    def test_counts_cover_categories_nobody_used(self) -> None:
        report = compare_labels({"a": "supported"}, {"a": "supported"}, categories=CATEGORIES)
        assert report.left_counts["unsupported"] == 0
        assert report.confusion["unsupported"]["supported"] == 0


class TestKappa:
    def test_is_undefined_when_one_category_is_used_exclusively(self) -> None:
        """Expected agreement is 1.0, so there is no chance-corrected signal."""
        report = compare_labels(
            {"a": "supported", "b": "supported"},
            {"a": "supported", "b": "supported"},
            categories=CATEGORIES,
        )
        assert report.raw_agreement == 1.0
        assert report.kappa is None

    def test_perfect_agreement_across_two_categories_is_one(self) -> None:
        report = compare_labels(
            {"a": "supported", "b": "unsupported"},
            {"a": "supported", "b": "unsupported"},
            categories=CATEGORIES,
        )
        assert report.kappa == pytest.approx(1.0)

    def test_total_disagreement_is_negative(self) -> None:
        report = compare_labels(
            {"a": "supported", "b": "unsupported"},
            {"a": "unsupported", "b": "supported"},
            categories=CATEGORIES,
        )
        assert report.kappa is not None
        assert report.kappa < 0

    def test_skewed_marginals_keep_kappa_low_despite_high_raw_agreement(self) -> None:
        """The paradox the evaluation has to explain, pinned as behaviour."""
        left = {f"i{n}": "supported" for n in range(19)} | {"i19": "partially_supported"}
        right = {f"i{n}": "supported" for n in range(18)} | {
            "i18": "partially_supported",
            "i19": "supported",
        }
        report = compare_labels(left, right, categories=CATEGORIES)
        assert report.raw_agreement >= 0.9
        assert report.kappa is not None
        assert report.kappa < 0.2


class TestInterpretation:
    @pytest.mark.parametrize(
        ("kappa", "expected"),
        [
            (None, "undefined (one category used exclusively)"),
            (-0.1, "worse than chance"),
            (0.1, "slight"),
            (0.3, "fair"),
            (0.5, "moderate"),
            (0.7, "substantial"),
            (0.9, "almost perfect"),
        ],
    )
    def test_reports_the_conventional_reading(self, kappa: float | None, expected: str) -> None:
        assert interpret_kappa(kappa) == expected
