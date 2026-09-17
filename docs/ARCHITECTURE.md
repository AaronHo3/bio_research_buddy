# Architecture: Bio Research Copilot

Status: proposed. Update this file as implementation becomes real.

## Current system

Implemented as of Stage 2:

```text
bio-copilot search "<question>" --limit N --format json|table
bio-copilot ingest "<question>" --limit N [--db PATH]
bio-copilot show <source_id> [--source pubmed] [--db PATH]
  -> cli.py                     composition root: Settings -> httpx.Client, PubMedClient, SqliteRepository
  -> application/search.py      search_papers(source, question, max_results)
  -> application/ingest.py      ingest_papers(source, repository, ...) -> IngestReport manifest
  -> adapters/literature.py     LiteratureSource protocol + typed error hierarchy
  -> adapters/pubmed.py         esearch (PMIDs, relevance order) -> efetch (XML) -> Paper objects
  -> adapters/resilience.py     RetryPolicy (exponential backoff) + MinIntervalLimiter (NCBI pacing)
  -> adapters/persistence.py    PaperRepository protocol, UpsertResult, RetrievalRecord
  -> adapters/sqlite_repository.py  stdlib sqlite3; tables papers, retrievals, retrieval_papers
  -> domain/models.py           Paper, PublicationDate, SearchQuery (frozen Pydantic models)
  -> stdout (JSON or table); errors to stderr; exit 1 dependency failed, 2 bad input, 3 not found
```

Persistence: papers are keyed on `(source, source_id)` and upserted idempotently; `first_seen_at` is set once, `last_seen_at` refreshes. Each ingest run writes a `retrievals` row plus one `retrieval_papers` row per result with its rank, which is the provenance manifest. Authors are stored as JSON, partial dates as three nullable integers. Timestamps are ISO 8601 and must be timezone-aware. Schema version is tracked with `PRAGMA user_version`.

Observed in live use: PubMed's relevance sort is not stable across identical calls made seconds apart. Two consecutive ingests of the same question returned overlapping but different top-3 sets. Determinism holds for a given response, not for the live service; the evaluation set in Stage 3 must therefore be built from stored records, not re-fetched each run.

Boundary enforcement is tested: `domain/` imports no adapter code and `application/` imports only the port, never the PubMed module or an HTTP library. Unit tests replay recorded NCBI responses through `httpx.MockTransport`; a single `live`-marked test calls NCBI for real and is deselected by default.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried up to 3 times with 0.5 s, 1 s backoff; requests are paced to 3/s without an NCBI key and 10/s with one. Clock and sleep are injected so tests assert exact delays without waiting.

Not yet implemented: pagination, passages/chunking, embeddings, synthesis, API, UI.

## Initial vertical slice

```text
CLI query -> application service -> PubMed adapter -> normalized Paper objects -> terminal output
```

## Intended mature flow

```text
Browser -> API -> research service -> retrieval pipeline -> evidence set
                                      -> synthesis provider -> cited answer
                     -> PostgreSQL/pgvector
```

## Boundaries

- Domain models must not depend on PubMed response shapes.
- External literature and model providers sit behind adapters.
- Retrieval scoring and evaluation remain callable without the web UI.
- The API validates requests and maps domain errors to stable responses.

## Decision log

### ADR-001: Begin with a modular monolith

- Context: the product has one developer and uncertain requirements.
- Choice: one deployable application with clear internal modules.
- Alternatives: microservices or separate agent services.
- Consequence: simpler development and deployment; modules can separate later if measured scaling or ownership needs justify it.

### ADR-002: Deterministic retrieval before generation

- Context: end-to-end RAG failures are hard to diagnose.
- Choice: build and evaluate paper/evidence retrieval before adding an LLM.
- Consequence: slower path to a chat-like screenshot, stronger measurement and debuggability.

### ADR-003: Ports and adapters with an injected HTTP client

- Context: every later stage (persistence, evaluation, synthesis) consumes retrieved papers, and PubMed's XML shape must not leak into them. Tests must run offline.
- Choice: the application layer depends on a `LiteratureSource` protocol; `PubMedClient` implements it and receives an `httpx.Client` from the caller rather than creating one.
- Alternatives: call PubMed directly from the CLI; patch `httpx` at test time with `monkeypatch`; use a mocking library such as `respx`.
- Consequence: a fake source or a `MockTransport` replaces the network with no patching. Adding a second literature provider means one new adapter. The cost is one extra layer that is thin today.

### ADR-004: uv with a src layout

- Context: the spec left the environment manager open.
- Choice: `uv` for environment, lockfile, and Python version pinning; `src/` layout so tests import the installed package.
- Alternatives: Poetry, pip-tools, plain venv plus requirements files.
- Consequence: single fast tool with a committed lockfile; contributors need `uv` installed.

### ADR-005: SQLite behind a repository port before PostgreSQL

- Context: Stage 2 needs durable, idempotent storage of papers and retrieval provenance. The spec targets PostgreSQL with pgvector, but vector search is two stages away.
- Choice: stdlib `sqlite3` behind a `PaperRepository` protocol. No ORM, no server, no Docker yet.
- Alternatives: PostgreSQL from the start (Docker Compose, slower tests, setup before value); SQLAlchemy over SQLite for a configuration-only switch later (an ORM to learn now for a benefit not yet needed).
- Consequence: instant tests with `:memory:` databases and a single-file store. The Stage 4 move to PostgreSQL is a second repository implementation plus a data migration, which is a bounded, well-understood job.

## Open decisions

- Embedding and reranking models.
- LLM provider or local model.
- Frontend framework and deployment target.

