# Architecture: Bio Research Copilot

Status: proposed. Update this file as implementation becomes real.

## Current system

No implementation exists yet.

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

## Open decisions

- Python environment/package manager.
- Initial database introduction point.
- Embedding and reranking models.
- LLM provider or local model.
- Frontend framework and deployment target.

