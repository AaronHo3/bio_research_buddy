# Bio Research Copilot

[![CI](https://github.com/AaronHo3/bio_research_buddy/actions/workflows/ci.yml/badge.svg)](https://github.com/AaronHo3/bio_research_buddy/actions/workflows/ci.yml)

Status: Stage 2 complete. A tested command-line tool retrieves PubMed records, normalizes them into typed domain objects, and stores them in SQLite with provenance. No embeddings, language model, or web interface exist yet.

An evidence-grounded research assistant for biomedical literature. The application will retrieve relevant papers, synthesize a concise answer, and connect claims to inspectable evidence.

## Planned demo

Users will be able to enter a focused biomedical question, inspect retrieved evidence, and view a source-linked synthesis with explicit limitations.

## Engineering goals

- Evaluate keyword, semantic, hybrid, and reranked retrieval.
- Enforce traceable claim-to-evidence citations.
- Expose the system through a tested API and clean web interface.
- Deploy with reproducible infrastructure, monitoring, and cost controls.

## Usage

```bash
uv run bio-copilot search "CRISPR base editing off-target" --limit 3 --format table
```

```text
1. [24157548] Genome engineering using the CRISPR-Cas9 system.
   Nature protocols, 2013
   https://pubmed.ncbi.nlm.nih.gov/24157548/
2. [28375731] CRISPR-Cas9 Structures and Mechanisms.
   Annual review of biophysics, 2017
   https://pubmed.ncbi.nlm.nih.gov/28375731/
3. [25315507] Genome modification by CRISPR/Cas9.
   The FEBS journal, 2014
   https://pubmed.ncbi.nlm.nih.gov/25315507/
```

The default `--format json` prints the full normalized records, including abstracts, in a stable field order.

Store results locally and inspect them later:

```bash
uv run bio-copilot ingest "CRISPR base editing off-target" --limit 3   # prints a manifest
uv run bio-copilot show 24157548                                        # one stored paper
uv run bio-copilot passages 24157548                                    # its citable passages
uv run bio-copilot query "double nicking off-target" --k 3 --format table  # rank stored passages
uv run bio-copilot embed                                                # vectors for stored passages
uv run bio-copilot query "checkpoint inhibitor biomarkers" --method hybrid --rerank
```

MedCPT, NCBI's PubMed-trained retrieval stack, is an optional extra because it needs PyTorch:

```bash
uv sync --extra medcpt
uv run bio-copilot embed --model ncbi/MedCPT
uv run bio-copilot query "..." --method dense --model ncbi/MedCPT
uv run bio-copilot query "..." --rerank --rerank-model ncbi/MedCPT-Cross-Encoder
```

Ingesting is idempotent: re-running a question refreshes existing papers instead of duplicating them, and each run is recorded with the ranked IDs it returned. Each paper is also split into passages (title, then one per abstract section) with stable chunk IDs. The `query` command ranks those stored passages with a hand-written BM25 implementation, no network needed. That is the lexical baseline every later retrieval method has to beat. The database is a single SQLite file at `data/bio_copilot.sqlite` (git-ignored), configurable with `DATABASE_PATH` or `--db`.

Ask a question and get an answer in which every claim points at a passage you can open:

```bash
uv run bio-copilot answer "How does amyloid beta promote tau pathology?" --method hybrid
```

```text
1. Synergy between amyloid-β and tau in Alzheimer's disease.
   [pubmed:32778792:0]
2. Patients with Alzheimer's disease present with both extracellular amyloid-β plaques
   and intracellular tau-containing neurofibrillary tangles in the brain.
   [pubmed:32778792:1]

Sources
  pubmed:32778792:0  Synergy between amyloid-β and tau in Alzheimer's disease.
      https://pubmed.ncbi.nlm.nih.gov/32778792/
```

When the stored evidence does not support an answer, it says so rather than answering anyway.

By default, claims are sentences copied verbatim from the cited passage. That makes the answer unable to fabricate and equally unable to connect ideas across papers: a deliberate baseline. Synthesis by a language model is an optional extra that lifts that limit:

```bash
uv sync --extra llm
export ANTHROPIC_API_KEY=...        # or set it in .env
uv run bio-copilot answer "How does amyloid beta promote tau pathology?" --synthesis llm
```

The model is given each passage labelled with its real chunk id and must return claims through a schema in which the supporting ids are a required, non-empty field. Ids it was not given are dropped and counted, and the answer type still refuses to be constructed if any citation points at evidence the answer does not hold. Token counts and latency are reported on stderr, so `--format json` stays pipeable.

No API key is needed to use this project. Without one, `answer` runs the extractive provider and everything else works unchanged.

Exit codes: 1 the literature source or database failed, 2 the input was invalid, 3 the requested record is not stored.

For research and educational use only. Not medical advice.

## Current milestone

Stages 1 and 2 of the plan in `PROJECT_SPEC.md` are done: a typed PubMed adapter behind a provider-neutral interface with retry and rate pacing, idempotent SQLite persistence with retrieval provenance and passages, fixture-based offline tests, one marked live test, and a CLI. Stage 3 is done: a BM25 lexical baseline, a committed 60-paper benchmark with 12 questions and exhaustive relevance judgments, and measured results with an error analysis. The baseline reaches 0.837 recall and 0.826 nDCG at k=10, and its clearest failure is a vocabulary mismatch that semantic retrieval should close. Method, numbers, and limitations are in [docs/EVALUATION.md](docs/EVALUATION.md). Stage 4 compares six retrieval configurations on one benchmark: lexical BM25, two dense encoders, rank fusion, and two cross-encoder rerankers, one general and one trained on PubMed. No configuration wins everything, and the cheapest one still holds the best ranking at depth. The most useful result was a negative one that turned out to be about the pipeline rather than the model: a biomedical encoder looked clearly worse until it was fed documents in the format it was trained on, at which point it matched the best ranking in the project. Every number is in [docs/EVALUATION.md](docs/EVALUATION.md).

Stage 5 is under way. The end-to-end path now exists: a question retrieves evidence, synthesis produces claims, every claim carries the id of the passage it came from, and the answer type refuses to be constructed if a citation points at evidence it does not hold. Two providers sit behind one interface: a deterministic extractive baseline that cannot fabricate, and an optional hosted language model that can paraphrase and combine passages but inherits the same grounding guarantee. Both are now measured on the benchmark. The result that matters is not fluency but refusal: the extractive baseline answered two of the three questions the benchmark marks unanswerable, matching on shared words while missing the subject entirely, and both language models abstained on all three. Sonnet also cites relevant papers more often (0.775 against 0.663) while Haiku refuses a third of the answerable questions to buy its lower latency. Faithfulness, whether a claim follows from the passage it cites, still needs judgment and is the remaining Stage 5 gate.


## Evaluation

```bash
uv run bio-copilot load-corpus evaluation/benchmark_v1/corpus.xml --db data/benchmark_v1.sqlite
uv run bio-copilot evaluate evaluation/benchmark_v1 --k 10 --db data/benchmark_v1.sqlite
```

The benchmark is a fixed snapshot, so scores change only when the retriever changes. Method, metrics, and limitations are in [docs/EVALUATION.md](docs/EVALUATION.md).

## Development setup

Requires [uv](https://docs.astral.sh/uv/). It installs the pinned Python version and all dependencies from the lockfile.

```bash
uv sync
cp .env.example .env   # then fill in values; the file is git-ignored
```

Run the quality checks that CI enforces:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

Tests marked `live` call real external services and are skipped unless requested explicitly:

```bash
uv run pytest -m live
```
