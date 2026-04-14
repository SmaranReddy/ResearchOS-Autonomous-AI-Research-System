# ── Base: install Python deps ───────────────────────────────────────────────
# Build context must be the project root (.) so requirements.txt is reachable.
# See docker-compose.yml: build.context: .
FROM python:3.11-slim AS base

WORKDIR /app

# curl is needed by the Docker HEALTHCHECK in docker-compose (curl -f /health).
# Install it here once so both dev and prod images have it.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps first — this layer is cached and only rebuilt when
# requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# ── Dev: source is volume-mounted at runtime ────────────────────────────────
# --reload removed: unstable on Windows + Docker Desktop volume mounts
# (inotify events are unreliable via virtioFS/WSL2, causing uvicorn to exit on startup)
FROM base AS dev
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Prod: bake source into image; 2 workers, no reload ──────────────────────
# Worker count: 2 is a safe default for a memory-constrained container.
# Each uvicorn worker loads the embedding model (~500 MB) into its own process,
# so more workers = more RAM. Scale via horizontal replicas instead.
FROM base AS prod
COPY ./backend/ .
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
