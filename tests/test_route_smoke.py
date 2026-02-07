"""Portfolio smoke test for route planning."""

from pathlib import Path

from marshab.analysis.routing import plan_route
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.models import BoundingBox


def _configure_test_paths(monkeypatch, tmp_path: Path) -> None:
    """Configure isolated storage paths for smoke tests."""
    monkeypatch.setenv("MARSHAB_PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MARSHAB_PATHS__CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("MARSHAB_PATHS__OUTPUT_DIR", str(tmp_path / "output"))


def test_route_smoke_from_top_site(monkeypatch, tmp_path: Path):
    """Select top site then generate route with multiple waypoints."""
    monkeypatch.setenv("MARSHAB_DEMO_SEED", "42")
    _configure_test_paths(monkeypatch, tmp_path)

    pipeline = AnalysisPipeline()
    roi = BoundingBox(lat_min=40.0, lat_max=40.3, lon_min=180.0, lon_max=180.3)
    results = pipeline.run(roi=roi, dataset="mola", threshold=0.5)
    assert len(results.sites) >= 1
    assert results.dem is not None

    top_site = results.sites[0]
    dem = results.dem
    # Convert top-site lat/lon into approximate pixel target.
    target_row = int((roi.lat_max - top_site.lat) / (roi.lat_max - roi.lat_min) * (dem.shape[0] - 1))
    target_col = int((top_site.lon - roi.lon_min) / (roi.lon_max - roi.lon_min) * (dem.shape[1] - 1))
    target_row = max(0, min(dem.shape[0] - 1, target_row))
    target_col = max(0, min(dem.shape[1] - 1, target_col))

    # Choose a start pixel away from the target.
    start_row = 2 if target_row > dem.shape[0] // 2 else dem.shape[0] - 3
    start_col = 2 if target_col > dem.shape[1] // 2 else dem.shape[1] - 3

    route = plan_route(
        start=(start_row, start_col),
        end=(target_row, target_col),
        weights={
            "distance": 1.0,
            "slope_penalty": 10.0,
            "roughness_penalty": 5.0,
        },
        dem=dem,
        constraints={
            "max_slope_deg": 25.0,
            "max_roughness": 20.0,
            "enable_smoothing": False,
            "cliff_threshold_m": None,
        },
    )

    assert len(route.waypoints) > 2
