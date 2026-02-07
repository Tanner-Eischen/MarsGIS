"""Portfolio smoke test for analysis pipeline."""

from pathlib import Path

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.models import BoundingBox


def _configure_test_paths(monkeypatch, tmp_path: Path) -> None:
    """Configure isolated storage paths for smoke tests."""
    monkeypatch.setenv("MARSHAB_PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MARSHAB_PATHS__CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MARSHAB_PATHS__OUTPUT_DIR", str(tmp_path / "output"))


def test_pipeline_smoke_synthetic(monkeypatch, tmp_path: Path):
    """Run end-to-end site selection in deterministic synthetic mode."""
    monkeypatch.setenv("MARSHAB_DEMO_SEED", "42")
    _configure_test_paths(monkeypatch, tmp_path)

    pipeline = AnalysisPipeline()
    roi = BoundingBox(lat_min=40.0, lat_max=40.3, lon_min=180.0, lon_max=180.3)
    results = pipeline.run(roi=roi, dataset="mola", threshold=0.5)

    assert len(results.sites) >= 1
