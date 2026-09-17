"""Runtime settings, read from environment variables or a local ``.env`` file.

Only the composition root (the CLI today, the API later) should construct
``Settings``. Everything below it receives plain values, so no module deep in
the pipeline depends on how configuration is loaded.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ncbi_tool_name: str = "bio-research-copilot"
    ncbi_contact_email: str | None = None
    ncbi_api_key: str | None = None
    http_timeout_seconds: float = Field(default=30.0, gt=0)
