"""Tests for sentence splitting, including the abbreviations that break naive splits."""

import pytest

from bio_research_copilot.domain import split_sentences


def test_splits_on_terminal_punctuation() -> None:
    assert split_sentences("First one. Second one! Third one?") == [
        "First one.",
        "Second one!",
        "Third one?",
    ]


def test_single_sentence_without_a_boundary() -> None:
    assert split_sentences("No terminal punctuation here") == ["No terminal punctuation here"]


def test_empty_and_whitespace() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_collapses_internal_whitespace() -> None:
    assert split_sentences("Wrapped\n  text here.") == ["Wrapped text here."]


@pytest.mark.parametrize(
    "text",
    [
        "Editing was efficient, e.g. in HEK293 cells, across all replicates.",
        "Results agreed with Smith et al. and were reproducible.",
        "Rates differed vs. the control group in every run.",
        "Values are shown in Fig. 2 of the supplement.",
        "Approx. 30 colonies were counted per plate.",
        "Samples were incubated for 30 min. before analysis.",
    ],
)
def test_abbreviations_do_not_end_a_sentence(text: str) -> None:
    assert split_sentences(text) == [text]


def test_decimal_points_do_not_end_a_sentence() -> None:
    assert split_sentences("The difference was significant at 0.05 across groups.") == [
        "The difference was significant at 0.05 across groups."
    ]


def test_initials_do_not_end_a_sentence() -> None:
    assert split_sentences("Work by Ran F. Ann established the protocol.") == [
        "Work by Ran F. Ann established the protocol."
    ]


def test_abbreviation_followed_by_a_real_boundary() -> None:
    text = "Editing worked, e.g. in HEK293 cells. Off-target rates stayed low."
    assert split_sentences(text) == [
        "Editing worked, e.g. in HEK293 cells.",
        "Off-target rates stayed low.",
    ]


def test_closing_bracket_after_the_full_stop() -> None:
    text = "Rates were low (see below.) Further work is needed."
    assert len(split_sentences(text)) == 2


def test_no_text_is_lost() -> None:
    text = "One sentence here. Another with e.g. an abbreviation. A third at 0.01 significance."
    sentences = split_sentences(text)
    assert "".join(sentences).replace(" ", "") == text.replace(" ", "")
