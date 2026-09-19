# The demo image.
#
# Stateless at runtime: each question is answered from live PubMed, so there is
# no database to provision and nothing to keep in sync. The only data baked in is
# web/data/answers.json, the recorded example answers, which are static.
#
# ANTHROPIC_API_KEY is read from the environment and must be supplied as a secret
# by the host, never built into the image. Without it the container still serves:
# answers fall back to the deterministic extractive provider and the page says so.
#
# Spend is capped by DEMO_DAILY_TOKENS. Set it deliberately before exposing this
# publicly; the default is a ceiling, not a recommendation.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first, so editing source does not invalidate the cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra web --extra llm --no-dev

# The recorded example answers. Small, and the only state the container holds.
COPY web ./web

# Hosts inject the port they expect to reach; 8000 is only the local default.
ENV PORT=8000
EXPOSE 8000

# build_default_app reads settings and wires the real adapters. create_app takes
# a literature source as an argument and cannot be used as a factory here.
CMD ["sh", "-c", "uv run uvicorn --factory bio_research_copilot.web.app:build_default_app --host 0.0.0.0 --port ${PORT}"]
