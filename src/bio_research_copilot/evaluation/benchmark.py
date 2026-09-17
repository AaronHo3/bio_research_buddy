"""Loading a versioned benchmark: questions with human relevance judgments."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["mechanism", "comparison", "target", "unanswerable"]


class BenchmarkQuestion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    type: QuestionType
    question: str = Field(min_length=1)
    relevant: list[str] = Field(default_factory=list, description="Judged-relevant paper ids.")

    @property
    def judged(self) -> bool:
        """Unanswerable questions are judged by construction; others need at least one id."""
        return self.type == "unanswerable" or bool(self.relevant)

    @model_validator(mode="after")
    def _unanswerable_has_no_relevant(self) -> "BenchmarkQuestion":
        if self.type == "unanswerable" and self.relevant:
            msg = f"question {self.id} is unanswerable but lists relevant papers"
            raise ValueError(msg)
        return self


class Benchmark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1)
    judgment_unit: Literal["paper"]
    judgment_scale: str = ""
    judged_by: str = ""
    judged_at: str = ""
    notes: list[str] = Field(default_factory=list)
    questions: list[BenchmarkQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "Benchmark":
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            msg = "benchmark question ids must be unique"
            raise ValueError(msg)
        return self


def load_benchmark(path: Path) -> Benchmark:
    """Read ``questions.json`` from a benchmark directory or a direct file path."""
    file = path / "questions.json" if path.is_dir() else path
    return Benchmark.model_validate(json.loads(file.read_text(encoding="utf-8")))
