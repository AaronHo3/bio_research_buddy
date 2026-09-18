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

## Dense retrieval compared with the lexical baseline

Retriever `dense:BAAI/bge-small-en-v1.5`, same corpus, same judgments. Every passage is embedded once and stored; a query is embedded at search time and ranked by cosine similarity.

| Metric | BM25 k=5 | Dense k=5 | BM25 k=10 | Dense k=10 |
|---|---|---|---|---|
| precision@k | 0.600 | **0.667** | **0.467** | 0.444 |
| recall@k | 0.572 | **0.644** | **0.837** | 0.787 |
| MRR | **0.944** | 0.870 | **0.944** | 0.870 |
| nDCG@k | 0.760 | **0.785** | **0.826** | 0.776 |
| latency (ms) | **0.1** | 3.5 | **0.2** | 3.3 |

### Neither method wins

Comparing nDCG question by question at k=10, the lexical baseline is better on six and the dense retriever on three. At k=5 it is five against three with one tie. Dense leads on the aggregate at k=5 and trails it at k=10, so even the direction of the result depends on the cut-off.

This is the case for hybrid retrieval, arrived at by measurement rather than by assumption. Replacing BM25 with embeddings would have made this benchmark worse at k=10.

### Dense wins exactly where the baseline was diagnosed weakest

Comparison questions were the lexical baseline's worst type. They are the type dense improves most.

| Type, k=5 | BM25 nDCG | Dense nDCG |
|---|---|---|
| comparison | 0.592 | **0.741** |
| mechanism | **0.897** | 0.848 |
| target | **0.791** | 0.766 |

On the question contrasting combined CTLA-4 and PD-1 blockade against monotherapy, dense reaches perfect recall and a perfect ranking at k=5 where the baseline reaches two thirds. The prediction from the error analysis held: a method that represents meaning rather than word overlap handles "A compared with B" better than one that sums independent term matches.

### The vocabulary-mismatch prediction was half right

The error analysis predicted dense retrieval would recover the biomarker papers that never use the word "biomarker". At k=10 on that question:

| | Relevant found of 10 |
|---|---|
| BM25 | 5 |
| Dense | 5 |
| Either | 7 |

Dense did recover two of the three papers named in the analysis. It also lost two that the baseline found. The count is identical and the papers are not. So the hypothesis was right about the mechanism and wrong about the outcome: semantic similarity reaches vocabulary-mismatched papers and gives up others in exchange.

### Union recall is the argument for fusion

Taking the union of both retrievers' top ten:

| Retriever | mean recall@10 |
|---|---|
| BM25 | 0.837 |
| Dense | 0.787 |
| Either | 0.890 |

The union beats both, and it beats the better of the two by more than the gap between them. The two methods fail on different questions, which is precisely the condition under which fusing rankings helps. That number, not a preference for the technique, is the reason to build hybrid retrieval next.

### Abstention survives the switch

Dense retrieval has no concept of "no match": every passage has some cosine similarity to every query, so it always returns a full page of results. The zebrafish question that returned nothing under BM25 returns ten papers under dense. The separation by score nevertheless holds.

| Retriever | Lowest answerable | Highest unanswerable |
|---|---|---|
| BM25 | 8.202 | 4.140 |
| Dense | 0.845 | 0.657 |

Both remain separable on this benchmark, so an abstention threshold stays viable for either. The thresholds are not interchangeable, since the scores are on unrelated scales.

### Cost

Dense retrieval is about sixteen times slower per query, and both are far below anything a user would notice. The dense figure includes embedding the query, which is the dominant term; the similarity computation itself is one matrix multiply. The real cost is the one-time embedding of the corpus, about eight seconds for 135 passages.

## Hybrid retrieval

Retriever `rrf(bm25-v1+dense:BAAI/bge-small-en-v1.5)`. Both rankings are read 50 deep and combined by reciprocal rank fusion with the conventional constant of 60.

| Metric | BM25 | Dense | Hybrid |
|---|---|---|---|
| precision@10 | 0.467 | 0.444 | **0.478** |
| recall@10 | 0.837 | 0.787 | **0.846** |
| MRR | **0.944** | 0.870 | 0.870 |
| nDCG@10 | **0.826** | 0.776 | 0.812 |
| nDCG@5 | 0.760 | **0.785** | 0.759 |

### Fusion improves coverage, not ordering

Hybrid takes the best precision and the best recall at k=10 and does not take the best ranking quality. Reciprocal rank fusion rewards agreement between retrievers. When both agree, the result is confidently placed. When they disagree, which is exactly when one of them is right and the other is wrong, fusion splits the difference and promotes a compromise.

That shows up most clearly in MRR, which lands on dense retrieval's 0.870 rather than the baseline's 0.944. Fusion inherits the worse method's mistakes at rank one without inheriting the better method's successes there. It has no way to know which retriever to trust on a given query.

Counting nDCG wins per question at k=10: the lexical baseline takes five, hybrid two, dense two.

This is the argument for reranking, and it is why production retrieval stacks put a reranker after fusion rather than treating fusion as the finished article. Fusion assembles a better candidate set; something else has to order it.

### Hybrid does not reach the union ceiling

| | mean recall@10 |
|---|---|
| BM25 | 0.837 |
| Dense | 0.787 |
| Hybrid | 0.846 |
| Either, unfused | 0.890 |

The union of two top-ten lists holds up to twenty papers, and only ten survive fusion. The gap between 0.846 and 0.890 is the price of compressing them back into one list of the same length. Fusing deeper and reranking the result is the way to close it.

### Sensitivity to the fusion constant

The constant flattens the curve near the top: a larger value makes rank position matter less and agreement matter more. Varying it, with everything else fixed:

| Constant | recall@10 | nDCG@10 | MRR | nDCG@5 |
|---|---|---|---|---|
| 1 | 0.857 | 0.830 | 0.870 | 0.811 |
| 5 | 0.857 | 0.827 | 0.870 | 0.808 |
| 10 | 0.857 | 0.825 | 0.870 | 0.793 |
| 20 | 0.848 | 0.814 | 0.870 | 0.773 |
| 60 (default) | 0.846 | 0.812 | 0.870 | 0.759 |
| 200 | 0.846 | 0.811 | 0.870 | 0.759 |

Three things to read from this.

**Smaller is better on this corpus, consistently.** At a constant of one, hybrid beats every other configuration on every metric except MRR, including the lexical baseline's nDCG. The conventional value of 60 comes from work fusing much deeper candidate lists than 135 passages, where flattening the top is appropriate. Here it discards useful rank information.

**MRR does not move at all.** Every constant gives 0.870. Whatever the weighting, fusion picks the same thing first. This is the strongest evidence that the ranking problem is structural rather than a tuning problem, and that a reranker rather than a better constant is what fixes it.

**These numbers must not be used to pick a value.** Nine questions with no held-out split means a constant chosen from this table is fitted to this benchmark and would not be expected to transfer. The default is reported as the headline result for that reason. Selecting a constant honestly needs a larger question set split into development and test halves, which is the clearest argument for growing the benchmark.

## Reranking

Retriever `rerank:Xenova/ms-marco-MiniLM-L-6-v2(...)`. A cross-encoder reads the question and each candidate together and reorders a shortlist of 25. Unlike the other methods it cannot add or remove candidates, only reorder them.

| Metric | BM25 | Hybrid | + rerank |
|---|---|---|---|
| precision@10 | 0.467 | 0.478 | **0.478** |
| recall@10 | **0.837** | 0.846 | 0.829 |
| MRR | **0.944** | 0.870 | 0.852 |
| nDCG@10 | **0.826** | 0.812 | 0.782 |
| nDCG@5 | 0.760 | 0.759 | 0.747 |
| latency (ms) | **0.2** | 3.5 | 1142 |

### The reranker made ranking worse, and that prediction was wrong

The fusion analysis concluded that fusion assembles a candidate set and something else must order it, and named reranking as that something. Measured, this reranker orders it worse. nDCG at k=10 falls from 0.826 to 0.782 and MRR from 0.944 to 0.852. The hypothesis was reasonable and the result does not support it.

### It is not uniformly worse, and the split is informative

| Type, k=5 | BM25 nDCG | Dense nDCG | Reranked nDCG |
|---|---|---|---|
| mechanism | 0.897 | 0.848 | **0.925** |
| target | **0.791** | 0.766 | 0.766 |
| comparison | 0.592 | **0.741** | 0.549 |

Reranking produces the best mechanism-question ranking measured anywhere in this project, with a perfect reciprocal rank on all three. It also produces the worst comparison-question ranking measured anywhere, below the lexical baseline it was meant to improve.

The likely cause is domain. This cross-encoder is trained on web search: short queries against short passages of general text. The questions here are long and technical and the passages are dense biomedical abstracts. A question phrased as a direct information need ("what mechanisms cause X") resembles its training data; one that asks for two things to be weighed against each other does not. That is a testable explanation, and the obvious test is a cross-encoder trained on biomedical literature.

### The first stage stopped mattering

Reranking a lexical shortlist and reranking a fused shortlist produce identical scores on every metric. The shortlists are not identical: at depth 25 they differ by four to twelve passages per question. Those differences simply never reach the top ten after reranking, because they are passages the cross-encoder rates poorly.

At this corpus size a shortlist of 25 is nearly a fifth of all 135 passages, so both first stages already contain essentially every passage worth ranking. The measured advantage of fusion disappears the moment a reranker runs behind it. That result is a property of a small corpus and should not be expected to survive at scale, where a shortlist is a vanishing fraction of the whole.

### The abstention signal is dramatically better

| Retriever | Lowest answerable | Highest unanswerable | Gap |
|---|---|---|---|
| BM25 | 8.202 | 4.140 | 4.062 |
| Dense | 0.845 | 0.657 | 0.188 |
| Reranked | +2.857 | -5.170 | 8.027 |

The cross-encoder separates answerable from unanswerable questions by a sign change. Every answerable question scores positive and every unanswerable one scores negative, with no near misses. The other methods separate them only by a margin on an arbitrary scale.

This is the most useful thing reranking contributed. A stage that orders results worse but knows far better whether any of them are worth showing is still worth having, and abstention is a Stage 5 requirement.

### Cost

Roughly one second per query against a fifth of a millisecond for the lexical baseline, about four thousand times slower. Cost is one model call per candidate, so it scales with shortlist depth rather than corpus size. For a user-facing query that is noticeable but acceptable; for an evaluation sweep across many configurations it is the dominant cost.

## MedCPT: does a biomedical model beat a general one?

The reranking analysis blamed a domain mismatch: a model trained on short web-search queries, applied to long questions over biomedical abstracts. MedCPT tests that directly. NCBI trained it on PubMed search logs, on what biomedical researchers actually clicked after issuing a query, and it ships both a bi-encoder pair and a cross-encoder.

The answer differs for the two.

### The cross-encoder confirms the hypothesis

| Reranker over the lexical baseline | nDCG@5 | nDCG@10 | MRR | latency (ms) |
|---|---|---|---|---|
| none | 0.760 | **0.826** | **0.944** | 0.2 |
| general (ms-marco-MiniLM) | 0.747 | 0.782 | 0.852 | 1142 |
| biomedical (MedCPT) | **0.793** | 0.788 | 0.926 | 3607 |

The biomedical cross-encoder produces the best nDCG at k=5 measured anywhere in this project. It recovers almost all the MRR the general one destroyed, 0.926 against 0.852. The clearest movement is on the question type the general reranker handled worst:

| Comparison questions, nDCG@5 | |
|---|---|
| general reranker | 0.549 |
| biomedical reranker | **0.729** |

The domain explanation held. A cross-encoder trained on biomedical literature does the job the general one could not.

### The bi-encoder appears to refute it, and the appearance is misleading

| Dense retriever | recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| bge-small (general) | **0.787** | **0.776** | **0.870** |
| MedCPT (biomedical) | 0.753 | 0.697 | 0.731 |

Run through this pipeline, the domain-specific encoder is clearly worse. Taken at face value that says specialisation does not pay, which is the opposite of what the cross-encoder just showed.

It is an artifact of a decision made three stages earlier. MedCPT's article encoder is documented as taking a title and an abstract together as a pair. `SectionChunker` (ADR-006) stores the title as one passage and the abstract as another, so the encoder never sees a paper in the shape it was trained on. It scores a bare title against a question, and an abstract stripped of its title.

Ranking whole papers in the documented format, bypassing the chunker entirely:

| MedCPT bi-encoder | recall@10 | nDCG@10 | MRR |
|---|---|---|---|
| through this pipeline, chunked | 0.753 | 0.697 | 0.731 |
| whole papers, title and abstract paired | 0.747 | **0.786** | **0.944** |

Ranking quality jumps by 0.089 nDCG, and reciprocal rank rises from 0.731 to 0.944, matching the best figure in the project. Recall barely moves, which is consistent: recall depends on which papers surface at all, and the format affects how well they are ordered.

So most of MedCPT's apparent deficit was caused by feeding it text in a shape it was not built for. The chunking strategy, chosen for good reasons about citation granularity, has a measured cost that only appeared once a model with a specific input contract was tried.

### What this actually shows

- **Domain-specific models earn their place, where the interface lets them.** The cross-encoder, which takes a query and a text with no format assumptions, wins clearly. The bi-encoder, which expects a particular document shape, loses until it is given one.
- **A negative result about a model can be a finding about your own pipeline.** The first MedCPT number would have supported "biomedical embeddings are not worth it". That conclusion would have been wrong and would have closed off the better option.
- **Chunking is not a free choice.** ADR-006 traded a model's preferred input format for citation granularity without knowing it was making that trade. A chunking strategy that keeps the title with the abstract would suit MedCPT and would coarsen citations; both effects are now measurable.
- **Cost is real.** The biomedical cross-encoder takes about 3.6 seconds per query against 1.1 for the general one and a fifth of a millisecond for the lexical baseline. Model size drives both quality and latency here.

## Summary of everything measured

At k=10 unless stated, on benchmark v1, nine judged questions.

| Configuration | recall@10 | nDCG@10 | nDCG@5 | MRR | latency (ms) |
|---|---|---|---|---|---|
| BM25 (baseline) | **0.837** | **0.826** | 0.760 | **0.944** | **0.2** |
| Dense, bge-small | 0.787 | 0.776 | 0.785 | 0.870 | 3.3 |
| Dense, MedCPT (chunked) | 0.753 | 0.697 | 0.681 | 0.731 | 12.4 |
| Hybrid (RRF) | **0.846** | 0.812 | 0.759 | 0.870 | 3.5 |
| BM25 + general reranker | 0.829 | 0.782 | 0.747 | 0.852 | 1142 |
| BM25 + MedCPT reranker | 0.779 | 0.788 | **0.793** | 0.926 | 3607 |

No configuration wins everything. The lexical baseline still holds the best nDCG at k=10 and the best MRR, at a thousandth of the cost of anything else. Hybrid fusion holds the best coverage. The biomedical reranker holds the best ranking at k=5. A production choice would depend on whether the answer shows five results or ten, and on what a second of latency is worth.

### Target to beat

Coverage: 0.846 recall at k=10, against a 0.890 union ceiling. Ordering: 0.826 nDCG and 0.944 MRR at k=10, still held by the cheapest method in the project. The clearest untested lead is a chunking strategy that keeps a title with its abstract, which the MedCPT diagnostic suggests is worth about 0.089 nDCG to a model that expects it.

## Synthesis: does a language model beat the extractive baseline?

Everything above measures retrieval. This measures what is done with what was retrieved. Three providers, same BM25 retrieval, same eight-passage evidence budget, all 12 questions.

Citation *integrity* is deliberately not in the table. `ResearchAnswer` refuses to hold a citation pointing outside its own evidence, so integrity is 100% by construction for every provider and reporting it would be measuring a tautology. The column that replaces it counts fabrication *attempts*: evidence ids a provider returned that it was never given.

| | Extractive | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|
| Answered, of 9 answerable | 9 | 8 | 6 |
| Abstained, of 3 unanswerable | 1 | **3** | **3** |
| False answers | **2** | 0 | 0 |
| False abstentions | 0 | 1 | 3 |
| Cited papers judged relevant | 0.663 | **0.775** | 0.722 |
| Claims per answer | 5.0 | 5.0 | 3.7 |
| Fabrication attempts | 0 | 0 | 0 |
| Tokens (12 questions) | 0 | 38,290 in / 5,454 out | 29,781 in / 2,010 out |
| Mean latency | 6 ms | 4,617 ms | 1,705 ms |

### The headline result is about safety, not fluency

The expected argument for a language model was that it writes better claims, and it does. The more important result is that it refuses better.

The extractive baseline answered two of the three questions the benchmark marks unanswerable. Asked which malaria vaccine has the highest efficacy in children, it returned claims about checkpoint-inhibitor efficacy and faecal microbiota transplantation. Asked about statin side effects in elderly patients, it returned a lifestyle-intervention trial in non-demented elderly patients and a lung cancer abstract. It matched on *efficacy*, on *elderly*, on *common*, and had no way to notice that none of the papers were about the subject of the question.

Both language models abstained on all three. That is the failure mode that matters most for a biomedical tool: confidently answering a question the corpus cannot support is worse than answering nothing, and it is exactly what term overlap cannot detect.

### The cost of refusing well is refusing too often

Haiku abstained on three answerable questions to Sonnet's one, and produced 3.7 claims per answer against 5.0. It is roughly 2.7 times faster and uses a third of the output tokens, and it loses a third of the answerable questions. Both models refused q09, *which kinases phosphorylate tau in Alzheimer's disease* — a question the corpus does contain evidence for, so that refusal is a retrieval-plus-judgment failure worth examining rather than a model defect alone.

For a research assistant, a false abstention costs a user one question. A false answer costs them their trust in every other answer. On that asymmetry Sonnet is the right default, which is what ADR-015 chose before any of these numbers existed.

### Comparison questions are still the weakest type, everywhere

Citation precision on comparison questions is 0.467 extractive, 0.578 Sonnet, 0.500 Haiku — the worst of the four types under every provider, exactly as retrieval evaluation found at Stage 4. Better synthesis does not repair evidence that was not retrieved. This is the clearest demonstration in the project that the two stages fail independently.

### Zero fabrication attempts, and what that is not

Across 24 model calls, neither model returned a single evidence id it had not been given. That is a good sign and it is not a measured rate: zero events in 24 calls supports no estimate of how often it would happen at scale. The machinery that catches it is tested against synthetic fabrication in the unit suite, because the live runs never exercised it.

### Faithfulness is not measured here

Whether a claim actually follows from the passage it cites needs judgment and is not in this table. The evaluation writes every claim out with the full text it cites, so judging runs against a fixed artifact rather than a fresh generation that may word things differently. That is the next piece of work, and until it exists the project's central claim is supported by construction for the extractive provider and unverified for the language models.

## Faithfulness: does a claim follow from the passage it cites?

Citation correctness asks whether the cited paper was relevant to the question. Faithfulness asks something narrower: granted this passage, does this claim follow from it? A claim can cite an entirely relevant paper and still say more than that paper said.

A model judge scores every claim against its cited passages, with three verdicts. `partially_supported` exists because the interesting failure is rarely a flat contradiction; it is a claim that generalises past its evidence. The judge is shown the claim and the passages and deliberately *not* the question, so it cannot reward a claim for being a good answer.

Using a model to grade a model is a real weakness. It is handled below by a control and by human validation, not by assertion.

| | Extractive | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|
| Claims judged | 55 | 40 | 22 |
| Supported | 53 | 28 | 20 |
| Partially supported | 0 | 11 | 1 |
| Unsupported | 2 | 1 | 1 |
| Strict rate | 0.964 | 0.700 | **0.909** |
| Lenient rate | 0.964 | **0.975** | 0.955 |

### The control, and what it caught

The extractive provider is a positive control: its claims are sentences copied verbatim out of the passage they cite, so a judge that works must score them as supported. Splitting its 55 claims by kind is decisive.

- 37 claims that are one sentence inside a longer passage: **37 supported, 0 otherwise.** The judge is reliable where the task is well posed.
- 18 claims identical to their entire passage, all of them titles: **16 supported, 2 unsupported.**

Identical inputs, inconsistent verdicts. The judge's stated reason for both refusals is that a title restating itself carries no evidence, which is a defensible position — but it took that position 2 times out of 18. So the judge has an error rate of roughly 11% on this class and 0% off it, and the class exists only because ADR-006 chunking emits titles as standalone passages. A second measurement now points at that decision.

### Haiku is more faithful than Sonnet, and that is not a contradiction

Sonnet's strict rate is 0.700 against Haiku's 0.909, which inverts the obvious expectation. The lenient rates are nearly identical, 0.975 and 0.955, and the claim counts explain the gap: Sonnet made 40 claims and Haiku 22.

Sonnet is expansive. It writes more claims, combines more, and 11 of them reach past their evidence — adding a causal reading where the passage reports an association, or attaching a consequence the passage does not state. Haiku is conservative: it refuses more questions, writes fewer claims per answer, and the ones it writes stay close to the text.

This is the same tradeoff as the abstention table seen from the other side. Sonnet answers more and overreaches more; Haiku answers less and overreaches less. Neither is simply better, and a product decision here depends on whether a reader is better served by more coverage or by tighter claims.

### Both models made the same inference, and the judge rejected it

The single `unsupported` verdict for each model is the same claim, q04:2, on the same passage. The passage reports that inhibiting or knocking down PARP14 raises effector T cell infiltration and lowers regulatory T cells. Both models turned that into a statement about PARP14 itself: Sonnet wrote that PARP14 "enhances effector T cell suppression", Haiku that it "reduces effector T cell infiltration". Both have the direction right.

What they did is infer a mechanism from a knockdown experiment, which is routine scientific reading and still strictly more than the passage states. The passage establishes what happens when PARP14 is removed, not what PARP14 does when present. That is the definition of `partially_supported` in this scheme, so the judge reached the right concern and then applied the wrong label; its own recorded reason concedes the claim is "roughly consistent" before rejecting it.

So the most severe verdict the judge issued on either model is one it argued itself out of. That is a finding about the judge, not about the synthesis.

### Human validation: the judge is stricter than its author

20 of Sonnet's 40 claims were drawn with a fixed seed and labelled by hand before the judge's verdicts were read.

| | Human | Judge |
|---|---|---|
| supported | 19 | 13 |
| partially_supported | 1 | 6 |
| unsupported | 0 | 1 |

Raw agreement 0.700, expected by chance 0.632, Cohen's kappa **0.184**, which the conventional reading calls slight. Both numbers are needed: raw agreement alone flatters a labelling this skewed, and kappa alone collapses toward zero precisely because both labellers used one category for most items. The confusion matrix is what actually explains them.

All six disagreements run the same way: human `supported`, judge stricter. Not one runs the other way. That is a calibration difference, not noise, and reading the cases shows what the difference is.

- **q06:1** The claim says PD-1 and PD-L1 are promising biomarkers *because* their interaction is linked to progression and immune escape. The passage states the biomarker status and the interaction as separate sentences and never connects them causally. The judge is right; the claim inserts a connective the passage does not license.
- **q07:2** The cited passage is a bare title, "Synergy between amyloid-β and tau in Alzheimer's disease." The claim adds "rather than either protein acting entirely independently." The judge is right, and this is the title-only failure mode again.
- **q04:2** The judge's own reason concedes the claim is "roughly consistent" and then marks it unsupported. Here the human label is the better one.

The pattern is a threshold: the human judged whether the gist held, the judge judged whether every connective was licensed by the text. Neither standard is wrong, and the project needs the stricter one, because the whole point of an evidence-grounded tool is that a reader can check a claim against its source and find it there.

Two consequences. The strict rates above should be read as a lower bound under a demanding standard rather than as a defect rate. And a single labeller comparing against a single judge cannot separate "the judge is strict" from "the labeller is lenient"; only a second human could.

## Known limitations

- 12 questions is enough to catch gross regressions and to support an error analysis, not to claim statistically significant differences between retrievers.
- Single judge, so no inter-annotator agreement.
- No development and test split, so no hyperparameter (such as the fusion constant) can be selected without overfitting. Growing the question set is the prerequisite for tuning anything.
- Every configuration is compared on the same chunking strategy, which is not neutral between models. The MedCPT diagnostic shows a model with a specific input contract is penalised by it.
- Judging is exhaustive within a topic but assumes no cross-topic relevance.
- Titles and abstracts only; full-text relevance is out of scope.
- Synthesis results come from one run per provider. The SDK exposes no temperature control, so the numbers are not exactly reproducible; abstention decisions in particular could differ on a re-run.
- Faithfulness is unmeasured. Citation correctness says a cited paper was relevant to the question, not that the claim follows from it.
- Zero fabrication attempts in 24 calls is not evidence of a low fabrication rate.
- Faithfulness verdicts come from a model judge validated on 20 claims by one labeller, at kappa 0.184. Agreement is low and one-directional, so the strict rates are a lower bound under a demanding standard rather than a defect rate. A second labeller is the prerequisite for separating a strict judge from a lenient human.
- The judge and the synthesizer under test are the same model family, which may bias verdicts in a direction the control cannot detect.
