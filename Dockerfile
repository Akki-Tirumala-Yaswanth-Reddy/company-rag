FROM python:3.14.7-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

# Copy the uv binary from its official image (pin the version for reproducible builds)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

# 1. Install dependencies only. This layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 2. App code and data
COPY . .

# Put the project's venv first on PATH, so `python` and `uvicorn` come from it
ENV PATH="/app/.venv/bin:$PATH"
RUN python scripts/seed_db.py

COPY start.sh ./ 

# 3. Don't run as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
# Ingest at startup: the embedding API key only exists at runtime (--env-file), not at build time.
# Ingestion is idempotent, so restarts skip already-embedded chunks.
CMD ["/app/start.sh"]