# Architecture: Bio Research Copilot

Status: proposed. Update this file as implementation becomes real.

## Current system

Implemented as of Stage 2:

```text
bio-copilot search "<question>" --limit N --format json|table
bio-copilot ingest "<question>" --limit N [--db PATH]
bio-copilot show <source_id> [--source pubmed] [--db PATH]
bio-copilot passages <source_id> [--source pubmed] [--db PATH]
bio-copilot query "<question>" --k N [--format json|table] [--db PATH]
bio-copilot load-corpus <efetch.xml> [--db PATH]
bio-copilot judge <benchmark_dir> [--scope topic|pool] [--db PATH]
bio-copilot embed [--model NAME] [--batch-size N] [--db PATH]
bio-copilot evaluate <benchmark_dir> [--k N] [--out report.json] [--db PATH]
  -> cli.py                     composition root: Settings -> httpx.Client, PubMedClient, SqliteRepository
  -> application/search.py      search_papers(source, question, max_results)
  -> application/ingest.py      ingest_papers(source, repository, ...) -> upsert, chunk, record -> IngestReport
  -> application/query.py       query_passages(repository, question, k) -> ranked RetrievedEvidence
  -> application/evaluate.py    evaluate_retrieval(repository, benchmark, k) -> EvaluationReport
  -> evaluation/benchmark.py    Benchmark, BenchmarkQuestion, load_benchmark (questions.json)
  -> evaluation/metrics.py      precision@k, recall@k, MRR, nDCG@k (binary relevance)
  -> retrieval/base.py          Retriever protocol
  -> application/embed.py       embed_corpus(repository, model) -> EmbedReport, incremental
  -> adapters/embedding.py      EmbeddingModel protocol (separate passage and query methods)
  -> adapters/fastembed_model.py  local ONNX encoder, no PyTorch, offline after first download
  -> retrieval/tokenize.py      lower-case, alphanumeric split, stopwords, no stemming
  -> retrieval/bm25.py          BM25Index ("bm25-v1"): in-memory index built from stored passages
  -> adapters/literature.py     LiteratureSource protocol + typed error hierarchy
  -> adapters/pubmed.py         esearch (PMIDs, relevance order) -> efetch (XML) -> Paper objects
  -> adapters/resilience.py     RetryPolicy (exponential backoff) + MinIntervalLimiter (NCBI pacing)
  -> adapters/persistence.py    PaperRepository protocol, UpsertResult, RetrievalRecord
  -> adapters/sqlite_repository.py  stdlib sqlite3; tables papers, retrievals, retrieval_papers, passages, embeddings
  -> domain/models.py           Paper, PublicationDate, Passage, SearchQuery, RetrievedEvidence
  -> domain/chunking.py         Chunker protocol; SectionChunker ("section-v1"): title + one passage per abstract section
  -> stdout (JSON or table); errors to stderr; exit 1 dependency failed, 2 bad input, 3 not found
```

Persistence: papers are keyed on `(source, source_id)` and upserted idempotently; `first_seen_at` is set once, `last_seen_at` refreshes. Each ingest run writes a `retrievals` row plus one `retrieval_papers` row per result with its rank, which is the provenance manifest. Authors are stored as JSON, partial dates as three nullable integers. Timestamps are ISO 8601 and must be timezone-aware. Schema version is tracked with `PRAGMA user_version`.

Passages: each ingested paper is split by the named chunking strategy and its passages replace any stored earlier, so a shrunken abstract leaves no stale chunks. `chunk_id` is `source:source_id:position`, stable across runs; `content_hash` (SHA-256 of the text) changes when the text does, which later lets embeddings detect staleness. The strategy name is stored on every row.

Local retrieval: `query` tokenizes the question, builds a BM25 index over every stored passage, and returns ranked `RetrievedEvidence` carrying the passage, rank, score, per-method component scores, and the retriever name. The index is rebuilt per query, which is milliseconds at this corpus size and guarantees it never drifts from the store. Ties break on `chunk_id`, so output is deterministic for a given store.

Embeddings: vectors are stored per passage per model, as packed float32 blobs keyed on `(chunk_id, model)`. Each row carries the `content_hash` of the text it was computed from, so `embed` only sends passages that are new or whose text changed. Two models can coexist in one database, which is what makes a head-to-head comparison a single extra command rather than a second database.

Re-ingesting a paper updates its surviving passages in place instead of deleting and reinserting them, so their vectors survive; only a passage that genuinely disappears is deleted, and its vectors cascade away with it. Without that, every re-ingest would discard the whole corpus of vectors.

Evaluation: `evaluation/benchmark_v1/` holds a committed 60-paper PubMed snapshot, a manifest, and 12 questions with paper-level judgments. `load-corpus` stores the snapshot offline, `judge` prints pooled candidates for the human judge, and `evaluate` scores a retriever and writes a JSON report with per-question rankings. See `docs/EVALUATION.md`.

Observed in live use: PubMed's relevance sort is not stable across identical calls made seconds apart. Two consecutive ingests of the same question returned overlapping but different top-3 sets. Determinism holds for a given response, not for the live service; the evaluation set in Stage 3 must therefore be built from stored records, not re-fetched each run.

Boundary enforcement is tested: `domain/` imports no adapter code and `application/` imports only the port, never the PubMed module or an HTTP library. Unit tests replay recorded NCBI responses through `httpx.MockTransport`; a single `live`-marked test calls NCBI for real and is deselected by default.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried up to 3 times with 0.5 s, 1 s backoff; requests are paced to 3/s without an NCBI key and 10/s with one. Clock and sleep are injected so tests assert exact delays without waiting.

Not yet implemented: pagination, dense and hybrid retrieval over the stored vectors, synthesis, API, UI.

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

### ADR-006: One passage per abstract section, not fixed windows

- Context: passages are the unit of retrieval and citation. PubMed abstracts run 150 to 300 words and are sometimes labelled (BACKGROUND, RESULTS, ...).
- Choice: `SectionChunker` emits the title as one passage and then one passage per abstract section (the whole abstract when unlabelled). Deterministic positional ids, content hashes, and the strategy name are stored with each passage.
- Alternatives: sentence windows with overlap (finer citation targets, but a sentence splitter, near-duplicate chunks, and more rows for the same content); fixed character windows (fully general, but cuts mid-sentence and cites half-thoughts).
- Consequence: no sentence is split from its neighbours, and the evaluation in Stage 3 judges whole sections. If full text ever arrives, a windowed strategy is added as `window-v1` and both can coexist in the data because every passage names its strategy.

### ADR-007: Hand-written BM25 as the lexical baseline

- Context: Stage 3 needs a keyword baseline to measure semantic retrieval against. SQLite's FTS5 extension is available and would provide BM25 with an inverted index.
- Choice: implement BM25 directly (about sixty lines) over an in-memory index built from stored passages, with a deliberately simple tokenizer and no stemming.
- Alternatives: FTS5 (production-grade, scales, but a black box for evaluation writeups); a third-party BM25 package (a dependency for a formula that fits on one screen).
- Consequence: every term of the formula is testable and explainable, and the evaluation can attribute failures to tokenization, idf, or length normalisation. Scaling beyond tens of thousands of passages would justify FTS5 as a second `Retriever` implementation.

### ADR-008: Local ONNX embeddings rather than a hosted API

- Context: Stage 4 needs an embedding model. The realistic options are a hosted API, a local model through PyTorch, or a local model through ONNX.
- Choice: a local ONNX encoder (`fastembed`) running `BAAI/bge-small-en-v1.5`, 384 dimensions, a 67 MB model and roughly 60 MB of runtime. The provider sits behind an `EmbeddingModel` port with separate passage and query methods.
- Alternatives: a hosted API costs money per call, needs a key and a network at query time, and lets the provider change the model underneath a recorded evaluation result; PyTorch is roughly 2 GB installed and slows CI noticeably.
- Consequence: free, offline after the first download, and reproducible, so a benchmark number stays valid indefinitely. Embedding all 135 benchmark passages takes about 8 seconds. Adding NCBI's MedCPT for the domain comparison will require PyTorch, which will be an optional dependency group rather than a default one.
- The port splits passage and query embedding because retrieval models are often asymmetric: some prepend an instruction to queries, and MedCPT ships two separate encoders. A single method would force those providers to guess, and guessing wrong degrades retrieval silently.

## Open decisions

- Reranking model, and whether reranking is warranted at all.
- LLM provider or local model.
- Frontend framework and deployment target.

