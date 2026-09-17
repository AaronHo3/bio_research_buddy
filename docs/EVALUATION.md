# Evaluation

Retrieval is measured before generation is added, on a fixed benchmark, so that every later change to the retrieval stack has a number it must beat and an error analysis to explain.

## Benchmark v1

Location: `evaluation/benchmark_v1/`.

| File | Purpose |
|---|---|
| `manifest.json` | The three topics, the PubMed query used for each, the 20 PMIDs it returned, and the snapshot time. |
| `corpus.xml` | The 60 PubMed records as fetched, with reference lists and indexing metadata stripped. This is the corpus. It never changes within a version. |
| `questions.json` | 12 questions with paper-level relevance judgments. |

### Why a snapshot

PubMed's relevance ordering is not stable: identical searches seconds apart return different top results. A benchmark that re-fetched its corpus would drift. The committed XML fixes the corpus so that a metric change means the retriever changed, not the data.

### Topics and question types

Three topics, chosen for familiarity so judgments can be made with confidence: CRISPR base editing, immune checkpoint immunotherapy, and amyloid/tau in Alzheimer's disease. Each topic has one question of each type from the project specification:

- **mechanism**: how or why something happens.
- **comparison**: two interventions or hypotheses weighed against each other.
- **target**: which methods, biomarkers, or molecules are involved.

Three further **unanswerable** questions are biomedical but outside the corpus. They exist to measure how the system behaves when the right answer is "nothing here".

### Judging protocol

Judgments are made by the author, at paper level, on a binary scale: a paper is relevant if its title or abstract directly addresses the question. Passage-level judgments were considered and rejected; with one or two passages per paper they would double the work for almost no extra signal.

Candidates are pooled: `bio-copilot judge` runs the lexical retriever for each unjudged question and prints the top candidates with title and abstract snippet. The judge reads them and records the relevant PMIDs in `questions.json`. Pooling is the standard shortcut from information-retrieval evaluation and has a known bias: papers no retriever surfaces are never judged. When a second retriever is added its pool is judged too, and the union is used.

An answerable question with an empty `relevant` list is **unjudged**, not "nothing relevant". The evaluation lists such questions and excludes them from every mean.

## Metrics

All at cut-off `k`, paper level, binary relevance. Implemented in `src/bio_research_copilot/evaluation/metrics.py` with hand-computed tests.

- **precision@k**: fraction of the top k that is relevant.
- **recall@k**: fraction of all relevant papers found in the top k.
- **MRR**: mean over questions of 1 / rank of the first relevant paper.
- **nDCG@k**: discounted gain, so an early hit is worth more than a late one, normalised by the ideal ordering.
- **latency**: wall-clock milliseconds per query, retriever only.

Unanswerable questions have no relevant set, so ranking metrics are undefined for them. The report shows how many papers came back and the top score, which is the raw material for a later abstention threshold.

Passage rankings are collapsed to paper rankings by giving each paper the rank of its best passage.

## Running it

```bash
uv run bio-copilot load-corpus evaluation/benchmark_v1/corpus.xml --db data/benchmark_v1.sqlite
uv run bio-copilot judge evaluation/benchmark_v1 --db data/benchmark_v1.sqlite      # fill in questions.json
uv run bio-copilot evaluate evaluation/benchmark_v1 --k 10 --db data/benchmark_v1.sqlite \
    --out evaluation/results/bm25-v1_benchmark_v1.json
```

The JSON report holds every question's ranked list and per-question metrics, so an error analysis can be written from it without re-running anything.

## Results

Results are recorded here only after judgments exist. No numbers yet.

## Known limitations

- 12 questions is enough to catch gross regressions and to support an error analysis, not to claim statistically significant differences between retrievers.
- Single judge, so no inter-annotator agreement.
- Pooling bias as described above.
- Titles and abstracts only; full-text relevance is out of scope.
