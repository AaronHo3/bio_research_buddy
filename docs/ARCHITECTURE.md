# Architecture: Bio Research Copilot

Status: proposed. Update this file as implementation becomes real.

## Current system

Implemented as of Stage 2:

```text
bio-copilot search "<question>" --limit N --format json|table
bio-copilot ingest "<question>" --limit N [--db PATH]
bio-copilot show <source_id> [--source pubmed] [--db PATH]
bio-copilot passages <source_id> [--source pubmed] [--db PATH]
bio-copilot query "<question>" --k N [--method bm25|dense|hybrid] [--rerank] [--format json|table] [--db PATH]
bio-copilot answer "<question>" [--method ...] [--rerank] [--min-score N] [--db PATH]
bio-copilot load-corpus <efetch.xml> [--db PATH]
bio-copilot judge <benchmark_dir> [--scope topic|pool] [--db PATH]
bio-copilot embed [--model NAME] [--batch-size N] [--db PATH]
bio-copilot evaluate <benchmark_dir> [--k N] [--method bm25|dense|hybrid] [--rerank] [--out report.json] [--db PATH]
  -> cli.py                     composition root: Settings -> httpx.Client, PubMedClient, SqliteRepository
  -> application/search.py      search_papers(source, question, max_results)
  -> application/ingest.py      ingest_papers(source, repository, ...) -> upsert, chunk, record -> IngestReport
  -> application/query.py       query_passages(repository, question, k) -> ranked RetrievedEvidence
  -> application/answer.py      answer_question(...) -> ResearchAnswer, or an abstention
  -> synthesis/base.py          SynthesisProvider protocol
  -> synthesis/context.py       select_evidence: rank-ordered, within a character budget
  -> synthesis/extractive.py    ExtractiveSynthesizer ("extractive-v1"): claims copied verbatim
  -> domain/sentences.py        rule-based sentence splitter with a biomedical abbreviation list
  -> application/evaluate.py    evaluate_retrieval(repository, benchmark, k) -> EvaluationReport
  -> evaluation/benchmark.py    Benchmark, BenchmarkQuestion, load_benchmark (questions.json)
  -> evaluation/metrics.py      precision@k, recall@k, MRR, nDCG@k (binary relevance)
  -> retrieval/base.py          Retriever protocol
  -> application/embed.py       embed_corpus(repository, model) -> EmbedReport, incremental
  -> adapters/embedding.py      EmbeddingModel protocol (separate passage and query methods)
  -> adapters/fastembed_model.py  local ONNX encoder, no PyTorch, offline after first download
  -> retrieval/tokenize.py      lower-case, alphanumeric split, stopwords, no stemming
  -> retrieval/bm25.py          BM25Index ("bm25-v1"): in-memory index built from stored passages
  -> retrieval/dense.py         DenseIndex ("dense:<model>"): exact cosine similarity over stored vectors
  -> retrieval/fusion.py        ReciprocalRankFusion ("rrf(a+b)"): combines rankings by position
  -> retrieval/rerank.py        RerankingRetriever: cross-encoder reorders a shortlist, never adds to it
  -> adapters/reranking.py      Reranker protocol; fastembed_reranker.py runs a local ONNX cross-encoder
  -> adapters/medcpt.py         NCBI MedCPT bi-encoder pair and cross-encoder (optional extra)
  -> adapters/model_factory.py  routes a model name to its backend; MedCPT imported lazily
  -> adapters/literature.py     LiteratureSource protocol + typed error hierarchy
  -> adapters/pubmed.py         esearch (PMIDs, relevance order) -> efetch (XML) -> Paper objects
  -> adapters/resilience.py     RetryPolicy (exponential backoff) + MinIntervalLimiter (NCBI pacing)
  -> adapters/persistence.py    PaperRepository protocol, UpsertResult, RetrievalRecord
  -> adapters/sqlite_repository.py  stdlib sqlite3; tables papers, retrievals, retrieval_papers, passages, embeddings
  -> domain/models.py           Paper, PublicationDate, Passage, SearchQuery, RetrievedEvidence, Citation, ResearchAnswer
  -> domain/chunking.py         Chunker protocol; SectionChunker ("section-v1"): title + one passage per abstract section
  -> stdout (JSON or table); errors to stderr; exit 1 dependency failed, 2 bad input, 3 not found
```

Persistence: papers are keyed on `(source, source_id)` and upserted idempotently; `first_seen_at` is set once, `last_seen_at` refreshes. Each ingest run writes a `retrievals` row plus one `retrieval_papers` row per result with its rank, which is the provenance manifest. Authors are stored as JSON, partial dates as three nullable integers. Timestamps are ISO 8601 and must be timezone-aware. Schema version is tracked with `PRAGMA user_version`.

Passages: each ingested paper is split by the named chunking strategy and its passages replace any stored earlier, so a shrunken abstract leaves no stale chunks. `chunk_id` is `source:source_id:position`, stable across runs; `content_hash` (SHA-256 of the text) changes when the text does, which later lets embeddings detect staleness. The strategy name is stored on every row.

Local retrieval: `query` tokenizes the question, builds a BM25 index over every stored passage, and returns ranked `RetrievedEvidence` carrying the passage, rank, score, per-method component scores, and the retriever name. The index is rebuilt per query, which is milliseconds at this corpus size and guarantees it never drifts from the store. Ties break on `chunk_id`, so output is deterministic for a given store.

Embeddings: vectors are stored per passage per model, as packed float32 blobs keyed on `(chunk_id, model)`. Each row carries the `content_hash` of the text it was computed from, so `embed` only sends passages that are new or whose text changed. Two models can coexist in one database, which is what makes a head-to-head comparison a single extra command rather than a second database.

Re-ingesting a paper updates its surviving passages in place instead of deleting and reinserting them, so their vectors survive; only a passage that genuinely disappears is deleted, and its vectors cascade away with it. Without that, every re-ingest would discard the whole corpus of vectors.

Grounding is enforced in the type, not by a check a caller might forget. `ResearchAnswer` validates that every cited id appears in the evidence the answer itself carries, so an answer citing something absent cannot be constructed at all, whatever produced it. The answer also carries only the evidence synthesis actually saw, which closes the gap where a provider could cite a passage that was retrieved but budgeted out.

Abstention is a first-class outcome with three causes: retrieval found nothing, the top score fell below a configured floor, or synthesis could not form a claim. Each records its reason in the answer's limitations.

Retrieval methods share one `Retriever` protocol and are selected with `--method`, so the evaluation command scores any of them unchanged. A retriever is built from the repository per call. The lexical index reads passages; the dense index reads passages paired with their stored vectors and refuses to run on a partially embedded corpus, because silently searching part of it would understate recall.

Evaluation: `evaluation/benchmark_v1/` holds a committed 60-paper PubMed snapshot, a manifest, and 12 questions with paper-level judgments. `load-corpus` stores the snapshot offline, `judge` prints pooled candidates for the human judge, and `evaluate` scores a retriever and writes a JSON report with per-question rankings. See `docs/EVALUATION.md`.

Observed in live use: PubMed's relevance sort is not stable across identical calls made seconds apart. Two consecutive ingests of the same question returned overlapping but different top-3 sets. Determinism holds for a given response, not for the live service; the evaluation set in Stage 3 must therefore be built from stored records, not re-fetched each run.

Boundary enforcement is tested: `domain/` imports no adapter code and `application/` imports only the port, never the PubMed module or an HTTP library. Unit tests replay recorded NCBI responses through `httpx.MockTransport`; a single `live`-marked test calls NCBI for real and is deselected by default.

Transient failures (timeouts, connection errors, HTTP 429/5xx) are retried up to 3 times with 0.5 s, 1 s backoff; requests are paced to 3/s without an NCBI key and 10/s with one. Clock and sleep are injected so tests assert exact delays without waiting.

Not yet implemented: pagination, a language-model synthesis provider, synthesis evaluation (citation correctness and faithfulness), API, UI.

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
- Alternatives: a hosted API costs money per call, needs a key and a network at query time, and lets the provider change the model underneath a recorded evaluation result; PyTorch was assumed to be roughly 2 GB installed; measured, the CPU-only build is about 120 MB, so the real cost is model weights rather than the framework.
- Consequence: free, offline after the first download, and reproducible, so a benchmark number stays valid indefinitely. Embedding all 135 benchmark passages takes about 8 seconds. NCBI's MedCPT was later added for the domain comparison as the optional `medcpt` extra; see ADR-013.
- The port splits passage and query embedding because retrieval models are often asymmetric: some prepend an instruction to queries, and MedCPT ships two separate encoders. A single method would force those providers to guess, and guessing wrong degrades retrieval silently.

### ADR-009: Exact brute-force vector search rather than an approximate index

- Context: dense retrieval needs nearest-neighbour search over the stored vectors.
- Choice: normalise every stored vector once at construction, then score a query with a single matrix multiply against all of them. Exact, and about 3 ms per query at 135 passages, most of which is embedding the query rather than the search.
- Alternatives: an approximate index such as HNSW, or a dedicated vector database. Both trade recall for speed and add a dependency or a service.
- Consequence: recall is exact, so a benchmark number reflects the retrieval method and not an index's approximation error. Nothing is added to the deployment. When the corpus outgrows brute force, an approximate index becomes another `Retriever` and the benchmark measures what the approximation costs.

### ADR-010: Keep both retrievers rather than replacing the baseline

- Context: after measuring, dense retrieval beat the lexical baseline at k=5 and lost to it at k=10, winning three of nine questions on nDCG and losing six.
- Choice: keep both behind `--method` and fuse them next, rather than adopting either as the single retriever.
- Evidence: the union of the two retrievers' top ten reaches 0.890 mean recall against 0.837 for the better one alone. They fail on different questions, which is the condition under which fusion helps.
- Consequence: two indexes are built per query instead of one, and the fusion needs its own evaluation. Replacing the baseline with embeddings, the obvious move, would have made this benchmark worse at k=10.

### ADR-011: Reciprocal rank fusion rather than score normalisation

- Context: hybrid retrieval must combine a lexical ranking with a dense one. BM25 scores are unbounded and corpus-dependent (roughly 8 to 20 here); cosine similarity lives in [-1, 1]. They cannot be added directly.
- Choice: fuse on rank position. A passage scores `weight / (constant + rank)` summed over the retrievers that returned it, read 50 deep before fusing.
- Alternatives: normalise each retriever's scores per query and take a weighted sum. That makes a passage's score depend on which other passages came back, so adding one document to the corpus can reorder unrelated results. Learning a combination needs training data the project does not have.
- Consequence: scale-free and robust, at the cost of discarding magnitude. An overwhelming match and a marginal one are indistinguishable if they share a rank.
- Measured: fusion took the best precision and recall at k=10 and did not improve ranking quality. MRR stayed at dense retrieval's 0.870 rather than the baseline's 0.944, and did not move across any fusion constant tested. Fusion improves the candidate set and cannot order it, which is the case for adding a reranker rather than tuning the fusion further.

### ADR-012: Reranking is an opt-in stage, not part of the default path

- Context: fusion improved the candidate set without improving its order, which pointed at a cross-encoder reranker. A cross-encoder reads the query and a passage together in one pass, so unlike the bi-encoders it cannot precompute anything and costs one model call per candidate.
- Choice: `RerankingRetriever` wraps any other retriever and is enabled with `--rerank`, off by default. It reorders a shortlist and can never add to it, so the wrapped retriever's recall at the shortlist depth bounds what it can achieve.
- Measured, and against the prediction: this reranker made ranking worse. nDCG at k=10 fell from 0.826 to 0.782 and MRR from 0.944 to 0.852, at roughly four thousand times the latency. It gave the best mechanism-question ranking in the project and the worst comparison-question ranking, which points at a domain mismatch: the model is trained on short web-search queries, not long questions over biomedical abstracts.
- It did contribute one thing decisively. Answerable and unanswerable questions separate by a sign change rather than by a margin on an arbitrary scale, which is a far stronger basis for abstention than either retriever offers.
- Consequence: kept, off by default, and reported honestly as a negative result for ordering and a positive one for abstention. The next experiment is a cross-encoder trained on biomedical literature, which tests the domain explanation directly.

### ADR-013: MedCPT as an optional extra, and what it revealed about chunking

- Context: the general-purpose cross-encoder made ranking worse and the failure pattern pointed at a domain mismatch. NCBI's MedCPT, trained on PubMed search logs, tests that directly and supplies both a bi-encoder pair and a cross-encoder.
- Choice: `adapters/medcpt.py` behind the existing ports, installed with `uv sync --extra medcpt`. A name-based factory routes to it so PyTorch is imported only by someone who asks for MedCPT.
- Two protocol changes were needed. `EmbeddingModel` now declares a `similarity` of `cosine` or `dot`, because MedCPT is trained on the raw inner product and normalising its vectors would discard signal. The existing split between `embed_query` and `embed_passages`, added speculatively in ADR-008, turned out to be required rather than nice to have: MedCPT ships two different networks for the two sides.
- Measured, and it splits. The cross-encoder confirmed the domain hypothesis, taking the best nDCG at k=5 in the project (0.793) and lifting comparison questions from 0.549 to 0.729. The bi-encoder appeared to refute it, scoring 0.697 nDCG at k=10 against 0.776 for the general model.
- That appearance was an artifact of ADR-006. MedCPT's article encoder expects a title and abstract together; `SectionChunker` stores them as separate passages, so the encoder never saw a paper in its trained shape. Ranking whole papers in the documented format lifts nDCG to 0.786 and MRR to 0.944, matching the best in the project. Most of the deficit was the pipeline, not the model.
- Consequence: the chunking strategy now has a measured cost, not just a rationale. A strategy that keeps a title with its abstract would suit models with that contract and would coarsen citation granularity. Both effects can now be measured rather than argued.

### ADR-014: A deterministic extractive baseline before any language model

- Context: Stage 5 requires an answer whose every claim maps to retrieved evidence, and which abstains when evidence is missing. A language model is the obvious way to write such an answer and brings a provider choice, a cost, a key, and an unfalsifiable faithfulness question.
- Choice: build the whole grounding pipeline against a `SynthesisProvider` port with a deterministic extractive provider first. Each claim is a sentence copied verbatim from a retrieved passage, selected by overlap with the question.
- Rationale: an extractive summariser cannot fabricate, so faithfulness is a property of its construction rather than a measurement. It therefore pins the ceiling on faithfulness at complete and sets a low floor on answer quality, which is precisely what a language model must then beat: better answers without losing grounding. It also lets the citation schema, the evidence budget, the abstention paths, and the integrity rule all be built and tested with no provider, no key and no cost, exactly as the lexical baseline did for retrieval.
- The integrity rule lives in `ResearchAnswer` as a validator rather than in the service. A provider that invents a citation fails at construction, so the guarantee holds for a future language model without that model being trusted.
- Observed weakness, from real output: the baseline selects sentences by term overlap, so it returns sentences about the right topic that do not answer the question, and passages that are paper titles become claims that read as citations. Both are the expected shape of the floor rather than defects to fix here.
- Consequence: the provider decision comes to the author with the machinery already proven and a concrete baseline to measure against.

### ADR-015: Hosted Claude for synthesis, opt-in, with tool-use output

- Context: ADR-014 left a deterministic extractive baseline and a `SynthesisProvider` port. The baseline cannot combine two papers into one claim, cannot answer a comparison question, and turns title passages into claims that read as citations. Those are the capabilities a language model adds, and the specification also asks for cost and latency to be evaluated, which only exist as numbers against a metered provider.
- Choice: Anthropic's Claude behind the existing port, installed with `uv sync --extra llm` and selected with `answer --synthesis llm`. Extraction stays the default, so the repository runs with no API key and no paid service.
- Structured output over prose. The model must fill a declared schema, a list of `{claim, evidence_ids}`, and `tool_choice` forces that call. A response containing prose instead means the provider broke its contract, and the caller abstains rather than parsing free text.
- Passages are labelled with their real `chunk_id`, not with positions. A citation is therefore either a genuine id or an obvious error, with no index-to-id translation step that could silently go wrong.
- Grounding is enforced at two layers that are deliberately not the same. The adapter drops evidence ids that were not supplied and counts them, so one invented id degrades an answer rather than corrupting it. Whatever survives must still satisfy the `ResearchAnswer` validator, which cannot construct an answer citing evidence it does not hold. The adapter is best-effort cleanup; the type is the guarantee.
- Consequence, accepted: paraphrase is now allowed, so faithfulness stops being a property of construction and becomes something that must be measured. That measurement is the next slice, and the dropped-citation count is the raw material for a fabrication rate.
- Observed from the live API, and the reason the live tests exist: a real model abstaining returned its empty array re-encoded as a string, `{"claims": "{\"claims\": []}"}`. The API accepts that because the outer shape still matches the schema, so strict parsing turned a correct abstention into an error. The adapter now unwraps a re-encoded payload rather than refusing it. A forced schema constrains shape, not sanity, and only a real call reveals the difference.
- Consequence, unwelcome: the current SDK exposes no temperature control, so identical input can produce differently worded answers on two runs. Determinism stops at the prompt, which is a pure function pinned to a golden file. The SDK also vendors its own HTTP library rather than `httpx`, so this one adapter's injected client differs in type from every other adapter in the project.


## Open decisions

- Whether a chunking strategy that keeps a title with its abstract is worth the coarser citations; the MedCPT diagnostic suggests it is worth about 0.089 nDCG to a model that expects that format.
- Whether a smaller model is enough for synthesis. ADR-015 defaults to Sonnet; the cost and quality comparison against Haiku is unmeasured.
- Frontend framework and deployment target.

