"""BM25 lexical ranking over passages, implemented from the formula.

For a query term ``t`` and passage ``p``::

    idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    tf_part     = tf(t,p) * (k1 + 1) / ( tf(t,p) + k1 * (1 - b + b * len(p) / avg_len) )
    score(q, p) = sum over unique t in q of  idf(t) * tf_part

- ``idf`` rewards rare terms: a word in every passage scores near zero.
- ``k1`` controls term-frequency saturation: the 5th repeat helps less than the 2nd.
- ``b`` controls length normalisation: long passages are penalised for matching by bulk.

The index is built in memory from a list of passages. That is fine for a
corpus of thousands of passages, and it keeps this module free of storage.
"""

import math
from collections import Counter
from dataclasses import dataclass

from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery
from bio_research_copilot.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class BM25Params:
    k1: float = 1.5
    b: float = 0.75


class BM25Index:
    name = "bm25-v1"

    def __init__(self, passages: list[Passage], params: BM25Params | None = None) -> None:
        self._params = params or BM25Params()
        # Stable order so equal scores always tie-break the same way.
        self._passages = sorted(passages, key=lambda p: p.chunk_id)
        self._term_freqs: list[Counter[str]] = [Counter(tokenize(p.text)) for p in self._passages]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(self._passages)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._passages)

    def score(self, query_terms: list[str], index: int) -> float:
        tf = self._term_freqs[index]
        length = self._lengths[index]
        k1, b = self._params.k1, self._params.b
        norm = k1 * (1.0 - b + b * (length / self._avg_len if self._avg_len else 0.0))
        total = 0.0
        for term in set(query_terms):
            freq = tf.get(term)
            if not freq:
                continue
            total += self._idf[term] * (freq * (k1 + 1.0)) / (freq + norm)
        return total

    def search(self, query: SearchQuery) -> list[RetrievedEvidence]:
        terms = tokenize(query.raw)
        if not terms or not self._passages:
            return []
        scored = [
            (self.score(terms, i), passage)
            for i, passage in enumerate(self._passages)
            if any(term in self._term_freqs[i] for term in terms)
        ]
        # Highest score first; ties by chunk_id (already sorted) via stable sort.
        scored.sort(key=lambda item: -item[0])
        return [
            RetrievedEvidence(
                passage=passage,
                rank=rank,
                score=score,
                components={"bm25": score},
                method=self.name,
            )
            for rank, (score, passage) in enumerate(scored[: query.max_results], start=1)
        ]
