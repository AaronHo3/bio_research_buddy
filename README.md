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
```

Ingesting is idempotent: re-running a question refreshes existing papers instead of duplicating them, and each run is recorded with the ranked IDs it returned. The database is a single SQLite file at `data/bio_copilot.sqlite` (git-ignored), configurable with `DATABASE_PATH` or `--db`.

Exit codes: 1 the literature source or database failed, 2 the input was invalid, 3 the requested record is not stored.

For research and educational use only. Not medical advice.

## Current milestone

Stages 1 and 2 of the plan in `PROJECT_SPEC.md` are done: a typed PubMed adapter behind a provider-neutral interface with retry and rate pacing, idempotent SQLite persistence with retrieval provenance, fixture-based offline tests, one marked live test, and a CLI. Next is passage creation with stable chunk IDs, then the retrieval baseline and evaluation set (Stage 3).

See `PROJECT_SPEC.md` for scope. This README should evolve into the public recruiter/user-facing document as verified features and results exist.


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
