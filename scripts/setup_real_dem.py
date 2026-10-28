"""Acquire the verified Jezero CTX DEM without altering its native georeferencing."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marshab.analysis.portfolio import acquire_dem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/source/jezero_ctx_20m.tif"))
    args = parser.parse_args()
    print(acquire_dem(args.out).resolve())


if __name__ == "__main__":
    main()
