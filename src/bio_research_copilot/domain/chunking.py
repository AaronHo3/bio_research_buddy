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
