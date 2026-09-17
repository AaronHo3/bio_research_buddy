"""A small, deterministic tokenizer shared by indexing and querying.

Deliberately simple: lower-case, split on anything that is not a letter or
digit, drop a short list of English stopwords. No stemming, so "editing" and
"editor" are different terms. That is a measurable limitation, not an
oversight; the evaluation in Stage 3 is where it gets quantified.
"""

import re

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "which", "with", "what", "how", "why", "does", "do", "did", "can",
        "could", "when", "where", "who", "whom",
    )
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens with stopwords removed. Order preserved."""
    return [t for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if t not in STOPWORDS]
