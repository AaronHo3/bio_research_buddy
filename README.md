# Bio Research Copilot

Status: Stage 1 complete. A tested command-line tool retrieves PubMed records and normalizes them into typed domain objects. No embeddings, language model, database, or web interface exist yet.

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

The default `--format json` prints the full normalized records, including abstracts, in a stable field order. Exit code 1 means the literature source failed; exit code 2 means the input was invalid.

For research and educational use only. Not medical advice.

## Current milestone

Stage 1 of the plan in `PROJECT_SPEC.md` is done: a typed PubMed adapter behind a provider-neutral interface, fixture-based offline tests, one marked live test, and a deterministic CLI. Next is retry, backoff, and rate-limit awareness in the adapter, followed by persistence (Stage 2).

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
