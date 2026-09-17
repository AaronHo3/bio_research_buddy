# Architecture: Bio Research Copilot

Status: proposed. Update this file as implementation becomes real.

## Current system

Implemented as of Stage 1:

```text
bio-copilot search "<question>" --limit N --format json|table
  -> cli.py            composition root: loads Settings, builds httpx.Client and PubMedClient
  -> application/search.py   search_papers(source, question, max_results)
  -> adapters/literature.py  LiteratureSource protocol + typed error hierarchy
  -> adapters/pubmed.py      esearch (PMIDs, relevance order) -> efetch (XML) -> Paper objects
  -> adapters/resilience.py  RetryPolicy (exponential backoff) + MinIntervalLimiter (NCBI pacing)
  -> domain/models.py        Paper, PublicationDate, SearchQuery (frozen Pydantic models)
  -> stdout (deterministic JSON or table); errors to stderr with exit codes 1 (source) / 2 (input)
```

Boundary enforcement is tested: `domain/` imports no adapter code and `application/` imports only the port, never the PubMed module or an HTTP library. Unit tests replay recorded NCBI responses through `httpx.MockTransport`; a single `live`-marked test calls NCBI for real and is deselected by default.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried up to 3 times with 0.5 s, 1 s backoff; requests are paced to 3/s without an NCBI key and 10/s with one. Clock and sleep are injected so tests assert exact delays without waiting.

Not yet implemented: pagination, persistence, passages, embeddings, synthesis, API, UI.

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

## Open decisions

- Initial database introduction point.
- Embedding and reranking models.
- LLM provider or local model.
- Frontend framework and deployment target.

