"""Run the strict, credential-free measured-DEM portfolio example."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marshab.analysis.portfolio import run_demo  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/source/jezero_ctx_20m.tif"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output/portfolio_demo"))
    args = parser.parse_args()
    manifest = run_demo(args.source, args.output_dir)
    print(f"Real CTX DEM verified: {manifest['source']['sha256']}")
    print(f"Grid: {manifest['grid']['shape']}; candidates: {len(manifest['candidates'])}")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
