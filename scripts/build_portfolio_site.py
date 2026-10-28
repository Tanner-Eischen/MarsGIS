"""CLI compatibility wrapper for scientific site packaging."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marshab.analysis.site_package import (  # noqa: E402
    DOWNLOADS,
    LAYER_SPECS,
    build_package,
    digest,
    main,
    read_probes,
    rgba,
    route_pixels,
    verify_package,
)

__all__ = [
    "DOWNLOADS",
    "LAYER_SPECS",
    "build_package",
    "digest",
    "read_probes",
    "rgba",
    "route_pixels",
    "verify_package",
]

if __name__ == "__main__":
    main()
