"""SQLite implementation of :class:`PaperRepository` using only the standard library.

Design notes:

- Papers are keyed on ``(source, source_id)``. Re-ingesting refreshes mutable
  fields and ``last_seen_at`` but never touches ``first_seen_at``.
- Authors are stored as a JSON array; the partial publication date as three
  nullable integers, mirroring the domain model rather than inventing a date.
- Timestamps are stored as ISO 8601 text and must be timezone-aware.
- The schema is created on open with ``IF NOT EXISTS``. ``PRAGMA user_version``
  records the schema version so a migration step can be added when needed.
"""

import json
import sqlite3
from array import array
from datetime import datetime
from pathlib import Path
from typing import Self

from bio_research_copilot.adapters.persistence import (
    PersistenceError,
    RetrievalRecord,
    RetrievedPaperRef,
    UpsertResult,
)
from bio_research_copilot.domain import Paper, Passage, PassageEmbedding, PublicationDate, Source

# Bumped when tables change. So far every change is additive and handled by
# CREATE TABLE IF NOT EXISTS; a real migration step is needed once a table changes shape.
SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    source        TEXT    NOT NULL,
    source_id     TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    abstract      TEXT,
    authors       TEXT    NOT NULL,
    journal       TEXT,
    pub_year      INTEGER,
    pub_month     INTEGER,
    pub_day       INTEGER,
    url           TEXT    NOT NULL,
    first_seen_at TEXT    NOT NULL,
    last_seen_at  TEXT    NOT NULL,
    PRIMARY KEY (source, source_id)
);

CREATE TABLE IF NOT EXISTS retrievals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question     TEXT    NOT NULL,
    max_results  INTEGER NOT NULL,
    retrieved_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS retrieval_papers (
    retrieval_id INTEGER NOT NULL REFERENCES retrievals(id),
    source       TEXT    NOT NULL,
    source_id    TEXT    NOT NULL,
    rank         INTEGER NOT NULL,
    PRIMARY KEY (retrieval_id, rank),
    FOREIGN KEY (source, source_id) REFERENCES papers(source, source_id)
);

CREATE TABLE IF NOT EXISTS passages (
    chunk_id     TEXT    PRIMARY KEY,
    source       TEXT    NOT NULL,
    source_id    TEXT    NOT NULL,
    position     INTEGER NOT NULL,
    section      TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    strategy     TEXT    NOT NULL,
    FOREIGN KEY (source, source_id) REFERENCES papers(source, source_id)
);

CREATE INDEX IF NOT EXISTS passages_by_paper ON passages (source, source_id, position);

-- ON DELETE CASCADE matters: replace_passages deletes a paper's passages before
-- reinserting them, and a vector for a passage that no longer exists is meaningless.
-- Without the cascade that delete fails on the foreign key instead.
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id     TEXT    NOT NULL REFERENCES passages(chunk_id) ON DELETE CASCADE,
    model        TEXT    NOT NULL,
    content_hash TEXT    NOT NULL,
    dimension    INTEGER NOT NULL,
    vector       BLOB    NOT NULL,
    PRIMARY KEY (chunk_id, model)
);

CREATE INDEX IF NOT EXISTS embeddings_by_model ON embeddings (model);
"""

_UPSERT_PAPER = """
INSERT INTO papers (
    source, source_id, title, abstract, authors, journal,
    pub_year, pub_month, pub_day, url, first_seen_at, last_seen_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (source, source_id) DO UPDATE SET
    title        = excluded.title,
    abstract     = excluded.abstract,
    authors      = excluded.authors,
    journal      = excluded.journal,
    pub_year     = excluded.pub_year,
    pub_month    = excluded.pub_month,
    pub_day      = excluded.pub_day,
    url          = excluded.url,
    last_seen_at = excluded.last_seen_at
"""

_SELECT_PAPER = """
SELECT source, source_id, title, abstract, authors, journal, pub_year, pub_month, pub_day, url
FROM papers WHERE source = ? AND source_id = ?
"""

_SELECT_PASSAGE = (
    "SELECT passages.chunk_id, passages.source, passages.source_id, passages.position,"
    " passages.section, passages.text, passages.content_hash, passages.strategy"
    " FROM passages"
)


class SqliteRepository:
    """A :class:`PaperRepository` backed by one SQLite file (or ``:memory:``)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @classmethod
    def open(cls, path: Path | str) -> Self:
        """Open or create the database at ``path``. ``":memory:"`` gives a throwaway store."""
        try:
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            return cls(sqlite3.connect(str(path)))
        except (sqlite3.Error, OSError) as exc:
            msg = f"could not open database at {path}: {exc}"
            raise PersistenceError(msg) from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- papers --------------------------------------------------------------

    def upsert_papers(self, papers: list[Paper], *, seen_at: datetime) -> UpsertResult:
        stamp = _stamp(seen_at)
        inserted = updated = 0
        with self._conn:
            for paper in papers:
                exists = self._conn.execute(
                    "SELECT 1 FROM papers WHERE source = ? AND source_id = ?",
                    (paper.source.value, paper.source_id),
                ).fetchone()
                date = paper.publication_date
                self._conn.execute(
                    _UPSERT_PAPER,
                    (
                        paper.source.value,
                        paper.source_id,
                        paper.title,
                        paper.abstract,
                        json.dumps(paper.authors, ensure_ascii=False),
                        paper.journal,
                        date.year if date else None,
                        date.month if date else None,
                        date.day if date else None,
                        paper.url,
                        stamp,
                        stamp,
                    ),
                )
                if exists:
                    updated += 1
                else:
                    inserted += 1
        return UpsertResult(inserted=inserted, updated=updated)

    def get_paper(self, source: Source, source_id: str) -> Paper | None:
        row = self._conn.execute(_SELECT_PAPER, (source.value, source_id)).fetchone()
        return _row_to_paper(row) if row else None

    def count_papers(self) -> int:
        count: int = self._conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return count

    # -- retrievals ----------------------------------------------------------

    def record_retrieval(
        self, question: str, max_results: int, papers: list[Paper], *, retrieved_at: datetime
    ) -> int:
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO retrievals (question, max_results, retrieved_at) VALUES (?, ?, ?)",
                    (question, max_results, _stamp(retrieved_at)),
                )
                retrieval_id = cursor.lastrowid
                if retrieval_id is None:
                    msg = "SQLite did not return a retrieval id"
                    raise PersistenceError(msg)
                self._conn.executemany(
                    "INSERT INTO retrieval_papers (retrieval_id, source, source_id, rank)"
                    " VALUES (?, ?, ?, ?)",
                    [
                        (retrieval_id, p.source.value, p.source_id, rank)
                        for rank, p in enumerate(papers, start=1)
                    ],
                )
        except sqlite3.IntegrityError as exc:
            msg = f"retrieval references a paper that is not stored: {exc}"
            raise PersistenceError(msg) from exc
        return retrieval_id

    def get_retrieval(self, retrieval_id: int) -> RetrievalRecord | None:
        header = self._conn.execute(
            "SELECT id, question, max_results, retrieved_at FROM retrievals WHERE id = ?",
            (retrieval_id,),
        ).fetchone()
        if header is None:
            return None
        rows = self._conn.execute(
            "SELECT source, source_id, rank FROM retrieval_papers"
            " WHERE retrieval_id = ? ORDER BY rank",
            (retrieval_id,),
        ).fetchall()
        return RetrievalRecord(
            id=header["id"],
            question=header["question"],
            max_results=header["max_results"],
            retrieved_at=datetime.fromisoformat(header["retrieved_at"]),
            papers=[
                RetrievedPaperRef(
                    source=Source(r["source"]), source_id=r["source_id"], rank=r["rank"]
                )
                for r in rows
            ],
        )

    # -- passages ------------------------------------------------------------

    def replace_passages(self, passages: list[Passage]) -> int:
        """Make the stored passages of each paper mentioned match the list exactly.

        Passages that survive are updated in place rather than deleted and
        reinserted, so their stored vectors survive too. Only a passage that
        genuinely disappears is deleted, taking its vectors with it. A surviving
        passage whose text changed gets a new ``content_hash``, which is what
        marks its vector stale without discarding it.
        """
        papers = sorted({(p.source.value, p.source_id) for p in passages})
        keep = {p.chunk_id for p in passages}
        try:
            with self._conn:
                for source, source_id in papers:
                    existing = self._conn.execute(
                        "SELECT chunk_id FROM passages WHERE source = ? AND source_id = ?",
                        (source, source_id),
                    ).fetchall()
                    self._conn.executemany(
                        "DELETE FROM passages WHERE chunk_id = ?",
                        [(r["chunk_id"],) for r in existing if r["chunk_id"] not in keep],
                    )
                self._conn.executemany(
                    "INSERT INTO passages (chunk_id, source, source_id, position,"
                    " section, text, content_hash, strategy)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT (chunk_id) DO UPDATE SET"
                    "   source = excluded.source, source_id = excluded.source_id,"
                    "   position = excluded.position, section = excluded.section,"
                    "   text = excluded.text, content_hash = excluded.content_hash,"
                    "   strategy = excluded.strategy",
                    [
                        (
                            p.chunk_id,
                            p.source.value,
                            p.source_id,
                            p.position,
                            p.section,
                            p.text,
                            p.content_hash,
                            p.strategy,
                        )
                        for p in passages
                    ],
                )
        except sqlite3.IntegrityError as exc:
            msg = f"passage references a paper that is not stored: {exc}"
            raise PersistenceError(msg) from exc
        return len(passages)

    def get_passages(self, source: Source, source_id: str) -> list[Passage]:
        rows = self._conn.execute(
            f"{_SELECT_PASSAGE} WHERE passages.source = ? AND passages.source_id = ?"
            " ORDER BY passages.position",
            (source.value, source_id),
        ).fetchall()
        return [_row_to_passage(r) for r in rows]

    def all_passages(self) -> list[Passage]:
        rows = self._conn.execute(f"{_SELECT_PASSAGE} ORDER BY passages.chunk_id").fetchall()
        return [_row_to_passage(r) for r in rows]

    def count_passages(self) -> int:
        count: int = self._conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        return count

    # -- embeddings ----------------------------------------------------------

    def upsert_embeddings(self, embeddings: list[PassageEmbedding]) -> int:
        try:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO embeddings (chunk_id, model, content_hash, dimension, vector)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT (chunk_id, model) DO UPDATE SET"
                    "   content_hash = excluded.content_hash,"
                    "   dimension    = excluded.dimension,"
                    "   vector       = excluded.vector",
                    [
                        (e.chunk_id, e.model, e.content_hash, e.dimension, _pack(e.vector))
                        for e in embeddings
                    ],
                )
        except sqlite3.IntegrityError as exc:
            msg = f"embedding references a passage that is not stored: {exc}"
            raise PersistenceError(msg) from exc
        return len(embeddings)

    def get_embeddings(self, model: str) -> list[PassageEmbedding]:
        rows = self._conn.execute(
            "SELECT chunk_id, model, content_hash, vector FROM embeddings"
            " WHERE model = ? ORDER BY chunk_id",
            (model,),
        ).fetchall()
        return [
            PassageEmbedding(
                chunk_id=r["chunk_id"],
                model=r["model"],
                content_hash=r["content_hash"],
                vector=_unpack(r["vector"]),
            )
            for r in rows
        ]

    def passages_needing_embedding(self, model: str) -> list[Passage]:
        """Missing vectors, plus vectors whose passage text has changed since embedding."""
        rows = self._conn.execute(
            f"{_SELECT_PASSAGE} LEFT JOIN embeddings e"
            "  ON e.chunk_id = passages.chunk_id AND e.model = ?"
            " WHERE e.chunk_id IS NULL OR e.content_hash != passages.content_hash"
            " ORDER BY passages.chunk_id",
            (model,),
        ).fetchall()
        return [_row_to_passage(r) for r in rows]

    def count_embeddings(self, model: str) -> int:
        count: int = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
        ).fetchone()[0]
        return count


def _pack(vector: list[float]) -> bytes:
    """Store a vector as packed float32. Compact, and exact enough for cosine similarity."""
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


def _stamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        msg = "timestamps must be timezone-aware"
        raise ValueError(msg)
    return moment.isoformat()


def _row_to_passage(row: sqlite3.Row) -> Passage:
    return Passage(
        source=Source(row["source"]),
        source_id=row["source_id"],
        chunk_id=row["chunk_id"],
        position=row["position"],
        section=row["section"],
        text=row["text"],
        content_hash=row["content_hash"],
        strategy=row["strategy"],
    )


def _row_to_paper(row: sqlite3.Row) -> Paper:
    date = None
    if row["pub_year"] is not None:
        date = PublicationDate(year=row["pub_year"], month=row["pub_month"], day=row["pub_day"])
    return Paper(
        source=Source(row["source"]),
        source_id=row["source_id"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors"]),
        journal=row["journal"],
        publication_date=date,
        url=row["url"],
    )
