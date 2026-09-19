# The demo image. Builds its own corpus at image-build time from the committed
# 449 KB PubMed XML, so the container needs no database service and no network
# at runtime. No API key is used or required: stored answers are static data and
# live answers use the deterministic extractive provider.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so edits to source do not invalidate the cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra web --no-dev

COPY evaluation/benchmark_v1 ./evaluation/benchmark_v1
COPY web ./web

# The SQLite corpus is not in version control; it is rebuilt from the XML.
RUN uv run bio-copilot load-corpus evaluation/benchmark_v1/corpus.xml \
      --db data/benchmark_v1.sqlite

EXPOSE 8000
ENV PORT=8000
CMD ["sh", "-c", "uv run uvicorn --factory bio_research_copilot.web.app:create_app --host 0.0.0.0 --port ${PORT}"]
