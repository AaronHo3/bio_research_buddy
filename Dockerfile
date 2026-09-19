# Container image for hosts that build from a Dockerfile, such as Hugging Face
# Spaces or Fly. Render does not need this: render.yaml uses Render's own Python
# runtime, which is one less moving part for a first deployment.
#
# Stateless at runtime. Each question is answered from live PubMed, so there is no
# database to provision and nothing to keep in sync. The only data baked in is
# web/data/answers.json, the recorded example answers.
#
# ANTHROPIC_API_KEY must be supplied as a secret by the host and never built into
# the image. Without it the container still serves: answers fall back to the
# deterministic extractive provider and the page says so. Spend is capped by
# DEMO_DAILY_TOKENS; set it deliberately before exposing this publicly.
#
# NOTE: this image has not been built. Docker was not installed on the machine
# this was developed on. The start command below is verified directly; the build
# is not.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Pinned to the version that produced uv.lock, so the lockfile is read by a tool
# that understands it. Installed with pip rather than copied from a registry
# image, which is one fewer external dependency at build time.
RUN pip install --no-cache-dir uv==0.12.5

WORKDIR /app

# Dependencies first, so editing source does not invalidate the cached layer.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra web --extra llm --no-dev

# The recorded example answers: small, and the only state the container holds.
COPY web ./web

# Hosts inject the port they expect to reach; 8000 is only the local default.
ENV PORT=8000
EXPOSE 8000

# build_default_app reads settings and wires the real adapters. create_app takes a
# literature source as an argument and cannot serve as a factory here.
CMD ["sh", "-c", "uv run uvicorn --factory bio_research_copilot.web.app:build_default_app --host 0.0.0.0 --port ${PORT}"]
