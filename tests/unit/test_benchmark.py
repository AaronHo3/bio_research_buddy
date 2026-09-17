"""Tests for benchmark loading, including the committed benchmark_v1 files."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bio_research_copilot.evaluation import Benchmark, load_benchmark, load_manifest

BENCHMARK_DIR = Path(__file__).parent.parent.parent / "evaluation" / "benchmark_v1"


def test_committed_benchmark_loads_and_is_well_formed() -> None:
    bench = load_benchmark(BENCHMARK_DIR)
    assert bench.version == "benchmark_v1"
    assert len(bench.questions) == 12
    assert sum(q.type == "unanswerable" for q in bench.questions) == 3
    assert {q.type for q in bench.questions} == {
        "mechanism",
        "comparison",
        "target",
        "unanswerable",
    }


def test_committed_manifest_and_corpus_agree() -> None:
    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
    pmids = [p for t in manifest["topics"].values() for p in t["pmids"]]
    assert len(pmids) == len(set(pmids)) == 60
    corpus = (BENCHMARK_DIR / "corpus.xml").read_text(encoding="utf-8")
    assert all(f'<PMID Version="1">{p}</PMID>' in corpus for p in pmids)


def test_judged_relevant_ids_must_exist_in_the_corpus() -> None:
    bench = load_benchmark(BENCHMARK_DIR)
    manifest = json.loads((BENCHMARK_DIR / "manifest.json").read_text(encoding="utf-8"))
    pmids = {p for t in manifest["topics"].values() for p in t["pmids"]}
    for q in bench.questions:
        assert set(q.relevant) <= pmids, f"{q.id} cites a paper outside the corpus"


def test_load_accepts_a_direct_file_path() -> None:
    assert load_benchmark(BENCHMARK_DIR / "questions.json").version == "benchmark_v1"


def test_manifest_loads_and_matches_the_questions() -> None:
    manifest = load_manifest(BENCHMARK_DIR)
    assert manifest.version == "benchmark_v1"
    assert len(manifest.topics) == 3
    assert all(len(t.pmids) == 20 for t in manifest.topics.values())
    assert len(manifest.paper_ids()) == 60
    topics = {q.topic for q in load_benchmark(BENCHMARK_DIR).questions if q.topic != "none"}
    assert topics == set(manifest.topics)


def test_manifest_accepts_a_direct_file_path() -> None:
    assert load_manifest(BENCHMARK_DIR / "manifest.json").version == "benchmark_v1"


def test_unanswerable_with_relevant_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unanswerable but lists"):
        Benchmark.model_validate(
            {
                "version": "t",
                "judgment_unit": "paper",
                "questions": [
                    {
                        "id": "x",
                        "topic": "t",
                        "type": "unanswerable",
                        "question": "?",
                        "relevant": ["1"],
                    }
                ],
            }
        )


def test_duplicate_ids_are_rejected() -> None:
    q = {"id": "x", "topic": "t", "type": "mechanism", "question": "?", "relevant": ["1"]}
    with pytest.raises(ValidationError, match="unique"):
        Benchmark.model_validate({"version": "t", "judgment_unit": "paper", "questions": [q, q]})


def test_unknown_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Benchmark.model_validate(
            {
                "version": "t",
                "judgment_unit": "paper",
                "questions": [{"id": "x", "topic": "t", "type": "vibes", "question": "?"}],
            }
        )


def test_judged_property() -> None:
    bench = load_benchmark(BENCHMARK_DIR)
    by_id = {q.id: q for q in bench.questions}
    assert by_id["q10"].judged  # unanswerable
    assert by_id["q01"].judged == bool(by_id["q01"].relevant)
