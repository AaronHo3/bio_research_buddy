# Project Specification: Bio Research Copilot

## Product statement

An evidence-grounded biomedical research assistant that finds relevant literature, synthesizes claims, exposes supporting evidence, and makes it easy to inspect the papers behind an answer.

This is not a generic “chat with PDFs” demo. It must demonstrate retrieval engineering, trustworthy citations, evaluation, API design, data persistence, deployment, and product judgment.

## Target users and jobs

- A biomedical researcher surveying a mechanism or target.
- A computational biologist comparing evidence across papers.
- A technically curious recruiter evaluating Aaron's AI engineering.

Core job: turn a focused research question into a traceable evidence brief while keeping sources visible.

## MVP user journey

1. User enters a biomedical research question.
2. System searches an approved literature source, initially PubMed.
3. System retrieves and ranks relevant records or passages.
4. System generates a concise synthesis only from supplied evidence.
5. Each meaningful claim links to supporting records/passages.
6. User can inspect title, abstract passage, authors, year, journal, PMID, and external link.
7. If evidence is missing or conflicting, the UI says so.

## Out of scope for MVP

- Clinical decision support or patient-specific advice.
- Paywalled full-text ingestion without rights.
- Exhaustive systematic reviews.
- Autonomous long-running web research.
- Multiple agent personas or complex orchestration.
- Authentication, teams, billing, or mobile apps.

## Architecture direction

Start as a modular monolith:

```text
Web UI
  -> FastAPI boundary
  -> application service
  -> query/retrieval pipeline
  -> literature + model provider adapters
  -> PostgreSQL/pgvector
  -> answer with evidence map
```

Suggested components, introduced only when needed:

- Python 3.12, `uv` or another documented environment manager.
- Pydantic for boundary/domain validation.
- `httpx` for external HTTP calls.
- FastAPI for public API and documentation.
- PostgreSQL plus pgvector after local retrieval needs persistence.
- A simple frontend that prioritizes evidence inspection.
- Docker Compose for reproducible local services.
- `pytest`, linting, type checking, and GitHub Actions.

Provider choices remain undecided until evaluated for cost, quality, licensing, and deployability.

## Domain model

Initial entities:

- `Paper`: stable source ID, title, abstract, authors, journal, publication date, URL, source.
- `Passage`: paper ID, text, section/location, deterministic chunk ID.
- `SearchQuery`: raw question and optional normalized terms.
- `RetrievedEvidence`: passage, component scores, rank, retrieval method.
- `Citation`: answer claim/span linked to evidence IDs.
- `ResearchAnswer`: question, answer text, evidence map, limitations, trace metadata.
- `EvaluationCase`: question, expected relevant source IDs or judged evidence, answer criteria.

## Implementation sequence and gates

### Stage 0: repository foundation

- Python package layout, configuration, lint/type/test commands, CI skeleton, `.env.example`, and public README scaffold.
- Gate: clean installation and one test in a fresh environment.

### Stage 1: literature retrieval slice

- Typed PubMed adapter with pagination, timeout, retry/backoff, rate-limit awareness, missing-field handling, and fixture-based tests.
- CLI query returns deterministic structured paper metadata.
- Gate: a real query and offline fixture tests both pass.

### Stage 2: persistence and ingestion

- Store normalized papers; define idempotent upsert and provenance.
- Add passage creation with stable chunk IDs and explicit strategy.
- Gate: rerunning ingestion creates no duplicates and produces an inspectable manifest.

### Stage 3: retrieval baseline and evaluation set

- Create a small versioned benchmark covering Aaron-relevant topics and unanswerable cases.
- Establish keyword/BM25-like baseline before semantic retrieval.
- Measure recall@k, precision@k or nDCG/MRR where judgments support them; report latency.
- Gate: evaluation command creates a reproducible report.

### Stage 4: semantic and hybrid retrieval

- Add embeddings, vector search, hybrid fusion, and optional reranking only after comparing baselines.
- Track embedding model/version and index provenance.
- Gate: justify the selected retrieval stack with measured results and error analysis.

### Stage 5: grounded synthesis

- Context builder applies token/length budget and preserves evidence IDs.
- Structured answer schema requires claim-to-evidence mapping.
- Unsupported questions abstain or qualify the response.
- Evaluate citation correctness, faithfulness, answer relevance, latency, and cost on the benchmark.
- Gate: no displayed citation can refer to absent evidence.

### Stage 6: API and product UI

- Validated search/research endpoints, consistent error format, health/readiness checks, and generated API docs.
- UI includes example questions, progress state, answer, evidence cards, source links, limitations, and error/empty states.
- Gate: a first-time user can run a guided query without instruction.

### Stage 7: production engineering

- Dockerized local environment, CI, deployment, structured logs, request IDs, caching/rate limiting where justified, cost limits, and basic monitoring.
- Gate: deploy from a clean commit, run smoke test, and document rollback/recovery.

### Stage 8: portfolio polish

- Architecture diagram, screenshots/GIF, results table, public demo, licenses, limitations, and evidence-backed résumé bullets.

## API sketch

- `GET /health`
- `POST /v1/search`: deterministic paper/evidence search.
- `POST /v1/research`: retrieval plus grounded synthesis.
- `GET /v1/papers/{source_id}`: stored metadata/evidence view.

Do not finalize schemas until the domain objects exist. Use versioned endpoints once public behavior stabilizes.

## Evaluation

Evaluate components separately:

### Retrieval

- Recall@k and rank-sensitive metric on judged questions.
- Performance by query type: mechanism, comparison, target, and unanswerable.
- Latency and provider failures.

### Generation

- Citation correctness: cited evidence supports the nearby claim.
- Citation completeness: supported claims are cited.
- Faithfulness: no material claim exceeds supplied context.
- Answer relevance and clarity.
- Abstention on insufficient evidence.

Human review remains important. Any model-based judge must be identified as a noisy evaluator, not ground truth.

## Safety, privacy, and responsible use

- Display “research and educational use, not medical advice.”
- Do not ingest user health information.
- Respect APIs, robots/rate limits, licenses, and redistribution terms.
- Avoid storing raw user questions by default in the public demo; if logged, disclose and minimize them.
- Escape/render model output safely.
- Protect paid APIs with quotas and server-side secrets.

## Failure cases to design for

- Missing abstracts, malformed dates, duplicate records, and retracted/updated papers.
- Query returns no results.
- External provider timeout or quota failure.
- Embedding/index version mismatch.
- Retrieval finds lexically similar but scientifically irrelevant evidence.
- LLM returns malformed output or an unsupported citation.
- Demo cold start and concurrent requests.

## Stretch goals

- Compare evidence between two hypotheses or interventions.
- Add metadata filters and saved evidence briefs.
- Detect retraction/erratum metadata where reliable.
- Add tool-driven query decomposition only if benchmark cases prove its value.
- Export a cited Markdown report.

## Portfolio completion evidence

- Public demo and repository.
- Versioned evaluation set and reproducible results.
- Baseline versus selected retrieval comparison.
- Grounding/citation error analysis.
- Measured latency and approximate per-query cost.
- README demo media and architecture.
- Résumé bullets based only on released results.

