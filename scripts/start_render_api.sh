#!/usr/bin/env bash
set -euo pipefail

# App code runs from /opt/render/project/src; orthophoto auto-discovery scans data/cache there.
mkdir -p /opt/render/project/src/data/cache

# Link the persistent-disk orthophoto into the app cache path.
if [[ -f /app/data/cache/hirise.tif ]]; then
  ln -sf /app/data/cache/hirise.tif /opt/render/project/src/data/cache/hirise.tif
fi

# Guard against missing runtime env injection by providing a safe default.
export MARSHAB_ORTHO_BASEMAP_PATH="${MARSHAB_ORTHO_BASEMAP_PATH:-/app/data/cache/hirise.tif}"

exec python -m marshab.web.server
