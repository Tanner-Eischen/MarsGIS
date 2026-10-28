# ruff: noqa: E402
"""CLI compatibility wrapper for independent scientific export verification."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marshab.analysis.verification import main  # noqa: E402
from marshab.analysis.verification import verify as verify

if __name__ == "__main__":
    main()
