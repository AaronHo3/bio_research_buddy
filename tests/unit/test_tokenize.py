"""Tests for the tokenizer: the one place indexing and querying must agree."""

from bio_research_copilot.retrieval import tokenize


def test_lowercases_and_splits_on_non_alphanumerics() -> None:
    assert tokenize("CRISPR-Cas9 off-target editing.") == [
        "crispr",
        "cas9",
        "off",
        "target",
        "editing",
    ]


def test_drops_stopwords_but_keeps_order() -> None:
    assert tokenize("What is the role of tau in Alzheimer's disease?") == [
        "role",
        "tau",
        "alzheimer",
        "s",
        "disease",
    ]


def test_keeps_unicode_letters() -> None:
    assert tokenize("Amyloid-β and GSK3β") == ["amyloid", "β", "gsk3β"]


def test_empty_and_punctuation_only() -> None:
    assert tokenize("") == []
    assert tokenize("... --- !!!") == []


def test_no_stemming_by_design() -> None:
    assert tokenize("editing editor edited") == ["editing", "editor", "edited"]
