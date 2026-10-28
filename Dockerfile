FROM node:22-bookworm-slim AS frontend

WORKDIR /build/webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ ./
ENV VITE_API_URL=/api/v1
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libexpat1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md marshab_config.yaml ./
COPY marshab/ ./marshab/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir --no-deps -e .

# Download the pinned 9.7 MB CTX source once at build time. acquire_dem verifies
# its SHA-256 and leaves the original planetary CRS/transform metadata intact.
RUN python -c "from pathlib import Path; from marshab.analysis.portfolio import acquire_dem; acquire_dem(Path('/app/reference/jezero_ctx20m.tif'))"
COPY --from=frontend /build/webui/dist/ ./webui/dist/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MARSHAB_WEB_HOST=0.0.0.0 \
    MARSHAB_WEB_PORT=5000 \
    MARSHAB_WEB_WORKERS=1 \
    WEB_CONCURRENCY=1 \
    MARSHAB_WEB_RELOAD=false \
    MARSHAB_WEBUI_DIST=/app/webui/dist \
    MARSHAB_PORTFOLIO_REFERENCE_DIR=/app/webui/dist/portfolio \
    MARSHAB_REFERENCE_DEM_PATH=/app/reference/jezero_ctx20m.tif \
    MARSHAB_PATHS__DATA_DIR=/app/data \
    MARSHAB_PATHS__CACHE_DIR=/app/data/cache \
    MARSHAB_PATHS__OUTPUT_DIR=/app/data/output \
    MARSHAB_MOLA_COG_BOOTSTRAP_MODE=skip \
    MARSHAB_MOLA_COG_REQUIRED=false \
    MARSHAB_PREWARM_COG=false \
    MARSHAB_TILE_ALLOW_DOWNLOAD=false \
    MARSHAB_ALLOW_SYNTHETIC_TILES=false \
    MARSHAB_ENABLE_EXPERIMENTAL_API=false \
    MARSHAB_CORS_ALLOW_ALL=false

RUN useradd --create-home --uid 10001 marshab \
    && mkdir -p /app/data/cache /app/data/output \
    && chown -R marshab:marshab /app/data
USER marshab
EXPOSE 5000
CMD ["python", "-m", "marshab.web.server"]
