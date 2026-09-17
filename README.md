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

