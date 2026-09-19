"""Turning a paper into citable passages.

Chunking is a pure function of the paper, so it lives in the domain layer. The
strategy is named and versioned; the name is stored with every passage so a
later change in chunking is visible in the data rather than silently mixed in.
"""

import hashlib
import re
from typing import Protocol

from bio_research_copilot.domain.models import Paper, Passage

# A structured-abstract section as the PubMed adapter renders it: an upper-case
# label, a colon, a space, then the body. Example: "RESULTS: We found ...".
_LABELLED_SECTION = re.compile(r"^([A-Z][A-Z0-9 /&,\-]{1,60}): (.+)$", re.DOTALL)

TITLE_SECTION = "title"
ABSTRACT_SECTION = "abstract"


class Chunker(Protocol):
    name: str

    def split(self, paper: Paper) -> list[Passage]: ...


class SectionChunker:
    """One passage for the title, then one per abstract section.

    PubMed abstracts are short (typically 150 to 300 words), so slicing them
    further would cut related sentences apart for no retrieval benefit. A
    structured abstract is split at its labelled sections; a plain abstract is
    one passage. Papers without an abstract yield only the title passage.
    """

    name = "section-v1"

    def split(self, paper: Paper) -> list[Passage]:
        pieces: list[tuple[str, str]] = [(TITLE_SECTION, paper.title)]
        if paper.abstract:
            for raw_part in paper.abstract.split("\n\n"):
                part = raw_part.strip()
                if not part:
                    continue
                match = _LABELLED_SECTION.match(part)
                if match:
                    pieces.append((match.group(1), match.group(2).strip()))
                else:
                    pieces.append((ABSTRACT_SECTION, part))
        return [
            Passage(
                source=paper.source,
                source_id=paper.source_id,
                chunk_id=f"{paper.source.value}:{paper.source_id}:{position}",
                position=position,
                section=section,
                text=text,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                strategy=self.name,
            )
            for position, (section, text) in enumerate(pieces)
        ]


FULL_TEXT_CHUNK_CHARS = 1400
"""Target characters per full-text chunk.

Short enough that a chunk is about one idea, so a claim citing it can be checked
against it quickly; long enough to keep a result and its qualification together,
which is where a shorter window loses meaning. A body of forty thousand
characters becomes roughly thirty chunks, which BM25 ranks in milliseconds.
"""

FULL_TEXT_OVERLAP_CHARS = 180
"""Carried from the end of one chunk into the next, so a sentence spanning a
boundary is whole in at least one of them."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_into_windows(text: str, *, size: int, overlap: int) -> list[str]:
    """Overlapping windows, broken at a sentence end where one is nearby.

    Breaking mid-sentence would produce a chunk whose first or last claim is
    unreadable, so the cut is pulled back to the last sentence boundary in the
    final quarter of the window when there is one.
    """
    if size < 1:
        msg = "size must be at least 1"
        raise ValueError(msg)
    if not 0 <= overlap < size:
        msg = "overlap must be at least 0 and smaller than size"
        raise ValueError(msg)

    cleaned = " ".join(text.split())
    if len(cleaned) <= size:
        return [cleaned] if cleaned else []

    windows: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            window = cleaned[start:end]
            boundary = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if boundary > int(size * 0.75):
                end = start + boundary + 1
        piece = cleaned[start:end].strip()
        if piece:
            windows.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return windows


class FullTextChunker:
    """Title, abstract sections, then the body split into overlapping windows.

    Full text is thirty times the length of an abstract, so it cannot be one
    passage: a citation has to point at something a reader can check without
    reading the paper. Sections are kept as the unit of meaning and split only
    when they exceed the window, with the section label carried onto every piece
    so a claim can say where in the paper it came from.

    Positions continue from the abstract chunker's, so a paper's passages share
    one numbering whether or not full text was available.
    """

    name = "fulltext-v1"

    def __init__(
        self,
        *,
        chunk_chars: int = FULL_TEXT_CHUNK_CHARS,
        overlap_chars: int = FULL_TEXT_OVERLAP_CHARS,
        abstract_chunker: Chunker | None = None,
    ) -> None:
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars
        self._abstract = abstract_chunker or SectionChunker()

    def split(self, paper: Paper) -> list[Passage]:
        """Abstract passages only. Full text arrives through :meth:`split_full_text`."""
        return self._abstract.split(paper)

    def split_full_text(self, paper: Paper, sections: list[tuple[str, str]]) -> list[Passage]:
        """Passages for a paper whose body was retrieved.

        Args:
            paper: The record the body belongs to.
            sections: ``(label, text)`` pairs in document order.
        """
        passages = list(self._abstract.split(paper))
        position = len(passages)
        for label, text in sections:
            for window in split_into_windows(
                text, size=self._chunk_chars, overlap=self._overlap_chars
            ):
                passages.append(
                    Passage(
                        source=paper.source,
                        source_id=paper.source_id,
                        chunk_id=f"{paper.source.value}:{paper.source_id}:{position}",
                        position=position,
                        section=label,
                        text=window,
                        content_hash=_sha(window),
                        strategy=self.name,
                    )
                )
                position += 1
        return passages
