"""Runtime settings, read from environment variables or a local ``.env`` file.

Only the composition root (the CLI today, the API later) should construct
``Settings``. Everything below it receives plain values, so no module deep in
the pipeline depends on how configuration is loaded.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ncbi_tool_name: str = "bio-research-copilot"
    ncbi_contact_email: str | None = None
    ncbi_api_key: str | None = None
    http_timeout_seconds: float = Field(default=30.0, gt=0)
    database_path: Path = Path("data/bio_copilot.sqlite")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = None
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # LLM synthesis is optional. Without a key the CLI still answers, using the
    # deterministic extractive provider, so the project runs with no paid service.
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = Field(default=2048, gt=0)
    judge_model: str = "claude-sonnet-5"
    # A question is not a PubMed query. Planning the translation is a short,
    # well-specified job, so the cheap fast model is the right one.
    planner_model: str = "claude-haiku-4-5-20251001"

    # Public demo guardrails. The daily token ceiling is the hard cap on what a
    # deployment can spend; at zero the demo answers extractively and costs nothing.
    demo_daily_tokens: int = Field(default=200_000, ge=0)
