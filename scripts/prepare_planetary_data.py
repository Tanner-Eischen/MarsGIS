"""Provision checksum-pinned global MOLA and detailed CTX sources before serving."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marshab.analysis.portfolio import acquire_dem  # noqa: E402
from marshab.config import get_config  # noqa: E402
from marshab.core.planetary_data import PlanetaryDataStore, ensure_global_dem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="Defaults to the configured cache directory"
    )
    args = parser.parse_args()
    output_dir = args.output_dir or get_config().paths.cache_dir
    global_path = ensure_global_dem(output_dir / "mola16.img")
    ctx_path = acquire_dem(output_dir / "jezero_ctx20m.tif")
    store = PlanetaryDataStore(global_path, ctx_path)
    print(f"Verified global MOLA: {global_path} ({store.global_provenance['sha256']})")
    print(f"Verified detailed CTX: {ctx_path} ({store.ctx_provenance['sha256']})")


if __name__ == "__main__":
    main()
