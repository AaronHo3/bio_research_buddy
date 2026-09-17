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
from datetime import datetime
from pathlib import Path
from typing import Self

from bio_research_copilot.adapters.persistence import (
    PersistenceError,
    RetrievalRecord,
    RetrievedPaperRef,
    UpsertResult,
)
from bio_research_copilot.domain import Paper, PublicationDate, Source

SCHEMA_VERSION = 1

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


def _stamp(moment: datetime) -> str:
    if moment.tzinfo is None:
        msg = "timestamps must be timezone-aware"
        raise ValueError(msg)
    return moment.isoformat()


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
