# Bio Research Copilot

Status: planning and repository setup.

An evidence-grounded research assistant for biomedical literature. The application will retrieve relevant papers, synthesize a concise answer, and connect claims to inspectable evidence.

## Planned demo

Users will be able to enter a focused biomedical question, inspect retrieved evidence, and view a source-linked synthesis with explicit limitations.

## Engineering goals

- Evaluate keyword, semantic, hybrid, and reranked retrieval.
- Enforce traceable claim-to-evidence citations.
- Expose the system through a tested API and clean web interface.
- Deploy with reproducible infrastructure, monitoring, and cost controls.

## Current milestone

Build a typed, tested PubMed retrieval client and CLI before introducing embeddings or an LLM.

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
