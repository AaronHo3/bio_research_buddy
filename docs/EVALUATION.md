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

Judging is **exhaustive**, not pooled. The corpus is 60 papers, and each question belongs to a topic of exactly 20, so every question can be judged against every paper that could plausibly answer it. `bio-copilot judge` prints one worksheet per topic: the topic's questions, then all 20 papers with title and abstract, each carrying a marker showing which questions currently list it.

```text
  marker slots, in order: q01 q02 q03

[.X.] 40119516  Recent advances in therapeutic gene-editing technologies.
      Gene editing has great therapeutic impact...
```

This matters because pooled judgments make recall circular. Round one used `--scope pool`, which showed only the lexical retriever's own top ten, so every judged paper was one that retriever had already found and recall@10 came out at exactly 1.000 by construction. `--scope pool` still exists for corpora too large to judge exhaustively, and it is the standard approach at scale, but it is not used here.

Cross-topic relevance is assumed absent: a question about tau kinases is judged against the Alzheimer's papers only. The topics were chosen to be disjoint, and the assumption is recorded here so a reader can challenge it.

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

Both precision and recall are bounded by how many papers are relevant to a question. Precision at k cannot exceed `min(k, relevant)/k`, and recall at k cannot exceed `min(k, relevant)/relevant`. Relevance counts here range from three to seventeen, so raw precision and recall are compared against those ceilings rather than against 1.0, and cross-type comparisons use nDCG and MRR, which normalise against the ideal ranking.

## Running it

```bash
uv run bio-copilot load-corpus evaluation/benchmark_v1/corpus.xml --db data/benchmark_v1.sqlite
uv run bio-copilot judge evaluation/benchmark_v1 --db data/benchmark_v1.sqlite      # fill in questions.json
uv run bio-copilot evaluate evaluation/benchmark_v1 --k 10 --db data/benchmark_v1.sqlite \
    --out evaluation/results/bm25-v1_benchmark_v1.json
```

The JSON report holds every question's ranked list and per-question metrics, so an error analysis can be written from it without re-running anything.

## Results

Retriever `bm25-v1`, benchmark `benchmark_v1`. Corpus of 60 papers and 135 passages, 9 judged questions, 61 judged-relevant papers from exhaustive per-topic judging. Full per-question reports are in `evaluation/results/`.

| Metric | k=5 | k=10 |
|---|---|---|
| precision@k | 0.600 | 0.467 |
| recall@k | 0.572 | 0.837 |
| MRR | 0.944 | 0.944 |
| nDCG@k | 0.760 | 0.826 |
| latency (ms) | 0.1 | 0.2 |

By question type:

| Type | n | P@5 | nDCG@5 | P@10 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| mechanism | 3 | 0.733 | 0.897 | 0.600 | 0.676 | 1.000 | 0.861 |
| target | 3 | 0.667 | 0.791 | 0.467 | 0.833 | 1.000 | 0.836 |
| comparison | 3 | 0.400 | 0.592 | 0.333 | 1.000 | 0.833 | 0.781 |

### Findings

**Recall is close to its achievable ceiling at k=10, and short of it at k=5.** The number of relevant papers per question ranges from three to seventeen, so a question with seventeen relevant papers cannot exceed 10/17 recall at k=10 no matter how good the retriever is. Measured against the achievable ceiling rather than against 1.0:

| | mean recall | mean ceiling | shortfall |
|---|---|---|---|
| k=5 | 0.572 | 0.801 | 0.229 |
| k=10 | 0.837 | 0.936 | 0.099 |

Six of the nine questions reach perfect recall at k=10. The shortfall is concentrated in two questions, and almost all of it in one.

**The clearest failure is a vocabulary mismatch, not a ranking problem.** Question six asks which biomarkers predict response to checkpoint inhibitors. Ten papers are relevant and BM25 returns five in its top ten. The three it ranks highest all use the word "biomarker" prominently. Three of the five it misses never use the word at all, and instead name a specific marker:

```text
MISSED  Targeting SRSF10 might inhibit M2 macrophage polarization and potentiate anti-PD-1 therapy
MISSED  Cell-intrinsic PD-L1 signaling drives immunosuppression by myeloid-derived suppressor cells
MISSED  Immune signature-based uncoupling of checkpoint inhibitor efficacy and toxicity
```

Each of these is a biomarker paper in substance and not in vocabulary. No amount of tuning the ranking formula finds them, because the query terms are absent from the text. This is the specific gap semantic retrieval should close, and question six is the case to watch when it is added.

**Raw precision cannot be compared across question types here.** Precision at k is capped at min(k, relevant)/k, and the count of relevant papers varies from three to seventeen across questions. At k=10 the comparison questions score 0.333, which is exactly their ceiling: they cannot do better because only three or four papers are relevant. Reporting that as a weakness would be wrong.

| Type | P@10 | ceiling | shortfall |
|---|---|---|---|
| mechanism | 0.600 | 0.767 | 0.167 |
| target | 0.467 | 0.633 | 0.167 |
| comparison | 0.333 | 0.333 | 0.000 |

nDCG and MRR normalise against the ideal ranking, so they do support cross-type comparison. Both are used below.

**Comparison questions are genuinely the weakest type, on the metrics that can say so.** Comparison scores 0.592 nDCG at k=5 against 0.897 for mechanism, and it holds the only reciprocal rank below 1.0. The retriever finds every relevant comparison paper by k=10, so this is a ranking failure rather than a retrieval failure. BM25 sums independent term matches and has no representation of one thing being weighed against another, so a paper that merely mentions both base editor types scores as well as one that compares them.

**Abstention on score alone looks feasible.** The lowest top score among answerable questions is 8.202, and the highest among unanswerable questions is 4.140. A threshold separates them cleanly here. Twelve questions cannot set a production threshold, but the signal is worth re-measuring as the corpus grows.

**Two of three unanswerable questions still return a full page of results.** Generic clinical vocabulary such as "patients", "efficacy", and "children" is common across the corpus, so keyword matching always finds something. Returning results is not the same as having evidence, which is what makes the score gap above matter.

**Latency is not a concern at this scale.** Well under a millisecond per query, so rebuilding the index on every call costs nothing worth optimising yet.

### Baseline to beat

Any retrieval change is measured against these numbers on this benchmark. The two specific targets are recall at k=10, currently 0.099 below its achievable ceiling, and nDCG on comparison questions, currently 0.592 at k=5.

## Known limitations

- 12 questions is enough to catch gross regressions and to support an error analysis, not to claim statistically significant differences between retrievers.
- Single judge, so no inter-annotator agreement.
- Judging is exhaustive within a topic but assumes no cross-topic relevance.
- Titles and abstracts only; full-text relevance is out of scope.
