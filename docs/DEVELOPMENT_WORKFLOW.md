# Development Workflow

This document describes how this project is built. It is the public record of the engineering process; the project log and architecture decision records hold the specifics.

## Principles

- **Work in small vertical slices.** Each slice delivers one end-to-end path (input to inspectable output) and is understood before the next begins. No stage introduces a component the previous stage did not prove necessary.
- **Define acceptance criteria before implementation.** Every task states what "done" means, which tests prove it, and how to verify it locally, before any code is written.
- **Test before merging.** Changes land only after the formatter, linter, type checker, and test suite pass. Unit tests use recorded fixtures; a separate, explicitly marked set of tests exercises real integrations.
- **Record architectural decisions.** Non-trivial choices are written as decision records with context, alternatives considered, and consequences. See `ARCHITECTURE.md`.
- **Evaluate against baselines.** No retrieval or generation component is adopted on intuition. A lexical baseline is measured first, and each addition (semantic search, hybrid fusion, reranking, synthesis) must justify itself on a versioned benchmark with reported metrics and error analysis.
- **Document limitations and failures.** Known failure modes, unanswerable cases, and measured weaknesses are recorded rather than hidden. Any model-based judge is treated as a noisy evaluator, not ground truth.
- **Verify every résumé claim with reproducible evidence.** A claim about performance, scale, or capability appears in public materials only when a committed evaluation script and report support it.

## Per-task loop

1. Inspect the current repository state and the relevant docs.
2. Define the task: purpose, concepts involved, proposed file changes, tests, and acceptance criteria.
3. Implement the smallest change that satisfies the criteria.
4. Run the configured checks (format, lint, type check, tests) and a manual smoke test.
5. Review the diff for secrets and unrelated edits.
6. Update documentation where implementation truth changed.
7. Append a project log entry.
8. Commit with a focused message describing one logical change.

## Branching and commits

- Feature work happens on short-lived branches tied to a single task or issue.
- Commits are small, self-contained, and explainable. A commit should not mix refactoring with behaviour changes.
- Generated artifacts, downloaded corpora, local indexes, and secrets are never committed. `.env.example` documents required configuration with placeholder values.

## Tooling

Development tooling, including editor and AI-assisted coding tools, is used for scaffolding, debugging, code review, and learning unfamiliar components. Design decisions, scope, evaluation methodology, and validation of every change remain the author's responsibility, and every committed line is one the author can explain and modify.

## Quality gates by stage

Each stage in `PROJECT_SPEC.md` ends with a gate. A stage is not complete until its gate is met and the result is recorded in `PROJECT_LOG.md`.
