"""Splitting a passage into sentences.

A citable claim should be a sentence, not an arbitrary span, so extractive
synthesis needs sentence boundaries. Biomedical abstracts break a naive split on
a full stop constantly: "e.g.", "et al.", "vs.", "approx. 3 mm", "Fig. 2",
"P. aeruginosa", and decimals such as "0.05" all contain a period that does not
end a sentence.

This is a small rule-based splitter with an explicit abbreviation list rather
than a statistical one. It is wrong on inputs the list does not cover, which is
a limitation worth stating rather than hiding: a mis-split produces an awkward
claim, never an unsupported one, because the text is still copied verbatim from
a passage that is still cited.
"""

import re

ABBREVIATIONS = frozenset(
    (
        # Latin and editorial. Stored without their internal periods, so "e.g."
        # and "eg" both normalise to the same entry.
        "eg", "ie", "etc", "cf", "vs", "al", "et", "viz", "ca", "approx",
        # Reference and figure
        "fig", "figs", "eq", "ref", "refs", "no", "nos", "vol", "pp", "ed", "eds",
        # Titles
        "dr", "prof", "mr", "mrs", "ms", "st",
        # Units and measures that appear mid-sentence
        "min", "sec", "hr", "hrs", "wk", "mo", "yr", "mg", "kg", "ml", "mm", "cm",
    )
)  # fmt: skip

# A boundary follows terminal punctuation and any closing quote or bracket, and
# precedes whitespace. Taking the closing bracket with the first sentence keeps
# "(see below.) Next" from starting a sentence with a stray bracket.
_BOUNDARY = re.compile(r"(?<=[.!?][\"')\]])(?=\s)|(?<=[.!?])(?=\s)")

# The final word before a trailing period, allowing internal periods so that
# "e.g." is captured whole rather than as its last letter.
_TRAILING_TOKEN = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)\.$")


def _ends_with_abbreviation(fragment: str) -> bool:
    stripped = fragment.rstrip().rstrip("\"')]")
    if not stripped.endswith("."):
        return False
    match = _TRAILING_TOKEN.search(stripped)
    if match is None:
        return False
    if match.group(1).replace(".", "").lower() in ABBREVIATIONS:
        return True
    # A single capital letter is an initial: "Ran F. Ann", "P. aeruginosa".
    final = match.group(1).split(".")[-1]
    return len(final) == 1 and final.isupper()


def _ends_mid_number(fragment: str, rest: str) -> bool:
    """A decimal point, as in '0.05', is never a sentence boundary."""
    stripped = fragment.rstrip()
    return bool(stripped) and stripped[-1] == "." and rest[:1].isdigit()


def split_sentences(text: str) -> list[str]:
    """Sentences in order, stripped, with empty ones dropped.

    Returns the whole text as one sentence when no boundary is found, so a
    caller always gets something citable back from non-empty input.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    pieces = _BOUNDARY.split(cleaned)
    sentences: list[str] = []
    buffer = ""
    for index, piece in enumerate(pieces):
        buffer += piece
        rest = pieces[index + 1].lstrip() if index + 1 < len(pieces) else ""
        if _ends_with_abbreviation(buffer) or _ends_mid_number(buffer, rest):
            continue
        trimmed = buffer.strip()
        if trimmed:
            sentences.append(trimmed)
        buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences
