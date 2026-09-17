"""Tests for the BM25 index against a tiny hand-built corpus.

Each test isolates one property of the formula so a regression names the term
that broke: idf (rarity), tf saturation, length normalisation, or ordering.
"""

import hashlib
import math

import pytest

from bio_research_copilot.domain import Passage, RetrievedEvidence, SearchQuery, Source
from bio_research_copilot.retrieval import BM25Index
from bio_research_copilot.retrieval.bm25 import BM25Params


def passage(chunk: str, text: str) -> Passage:
    return Passage(
        source=Source.PUBMED,
        source_id=chunk.split(":")[0],
        chunk_id=f"pubmed:{chunk}",
        position=int(chunk.split(":")[1]),
        section="abstract",
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        strategy="test",
    )


CORPUS = [
    passage("1:0", "tau phosphorylation drives tangle formation in neurons"),
    passage("2:0", "amyloid plaques accumulate in neurons before symptoms"),
    passage("3:0", "CRISPR base editing enables precise single base changes"),
    passage("4:0", "neurons neurons neurons neurons neurons neurons"),
]


def ids(results: list[RetrievedEvidence]) -> list[str]:
    return [r.passage.chunk_id for r in results]


def test_returns_only_passages_sharing_a_term() -> None:
    results = BM25Index(CORPUS).search(SearchQuery(raw="tau tangles"))
    assert ids(results) == ["pubmed:1:0"]


def test_no_overlap_gives_empty_result() -> None:
    assert BM25Index(CORPUS).search(SearchQuery(raw="zebrafish")) == []


def test_stopword_only_query_gives_empty_result() -> None:
    assert BM25Index(CORPUS).search(SearchQuery(raw="what is the")) == []


def test_empty_corpus_gives_empty_result() -> None:
    assert BM25Index([]).search(SearchQuery(raw="tau")) == []


def test_rare_term_outranks_common_term() -> None:
    """'neurons' is in 3 of 4 passages, 'amyloid' in 1: amyloid should dominate."""
    results = BM25Index(CORPUS).search(SearchQuery(raw="amyloid neurons"))
    assert ids(results)[0] == "pubmed:2:0"


def test_idf_of_a_term_in_every_passage_is_small_but_positive() -> None:
    corpus = [passage("1:0", "cell growth"), passage("2:0", "cell death")]
    index = BM25Index(corpus)
    assert index._idf["cell"] == pytest.approx(math.log(1 + 0.5 / 2.5))
    assert index._idf["growth"] > index._idf["cell"]


def test_term_frequency_saturates() -> None:
    """Six repeats of 'neurons' must not beat a passage where a rarer word also matches."""
    results = BM25Index(CORPUS).search(SearchQuery(raw="neurons tau"))
    assert ids(results)[0] == "pubmed:1:0"
    single_neurons = next(r for r in results if r.passage.chunk_id == "pubmed:2:0")
    six_neurons = next(r for r in results if r.passage.chunk_id == "pubmed:4:0")
    assert six_neurons.score < 2 * single_neurons.score


def test_length_normalisation_penalises_long_passages() -> None:
    short = passage("1:0", "kinase inhibitor")
    long_ = passage("2:0", "kinase inhibitor " + "filler " * 40)
    results = BM25Index([short, long_]).search(SearchQuery(raw="kinase"))
    assert ids(results) == ["pubmed:1:0", "pubmed:2:0"]
    with_b_zero = BM25Index([short, long_], BM25Params(b=0.0)).search(SearchQuery(raw="kinase"))
    assert with_b_zero[0].score == pytest.approx(with_b_zero[1].score)


def test_ranks_start_at_one_and_respect_max_results() -> None:
    results = BM25Index(CORPUS).search(SearchQuery(raw="neurons", max_results=2))
    assert [r.rank for r in results] == [1, 2]


def test_ties_break_by_chunk_id_deterministically() -> None:
    corpus = [passage("9:0", "same words here"), passage("2:0", "same words here")]
    results = BM25Index(corpus).search(SearchQuery(raw="words"))
    assert ids(results) == ["pubmed:2:0", "pubmed:9:0"]
    assert results[0].score == results[1].score


def test_evidence_carries_method_and_component_score() -> None:
    result = BM25Index(CORPUS).search(SearchQuery(raw="crispr"))[0]
    assert result.method == "bm25-v1"
    assert result.components == {"bm25": result.score}
    assert result.score > 0


def test_same_input_same_output() -> None:
    a = BM25Index(CORPUS).search(SearchQuery(raw="neurons amyloid tau"))
    b = BM25Index(list(reversed(CORPUS))).search(SearchQuery(raw="neurons amyloid tau"))
    assert a == b
