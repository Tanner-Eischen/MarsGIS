"""API tests for analysis dataset fallback behavior."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from marshab.core.analysis_pipeline import AnalysisResults
from marshab.exceptions import AnalysisError
from marshab.models import SiteCandidate
from marshab.web.api import create_app


def test_analyze_retries_with_mola_when_mola_200m_fails(monkeypatch, tmp_path):
    class StubPipeline:
        calls: list[str] = []

        def run(self, roi, dataset, threshold, criteria_weights=None, progress_callback=None):
            StubPipeline.calls.append(dataset)
            if dataset == "mola_200m":
                raise AnalysisError("Analysis pipeline failed", details={"error": "mola_200m unavailable"})

            site = SiteCandidate(
                site_id=1,
                geometry_type="POINT",
                area_km2=1.0,
                lat=18.3,
                lon=77.3,
                mean_slope_deg=2.0,
                mean_roughness=0.2,
                mean_elevation_m=-2500.0,
                suitability_score=0.91,
                rank=1,
            )
            return AnalysisResults(
                sites=[site],
                top_site_id=1,
                top_site_score=0.91,
            )

    monkeypatch.setattr("marshab.web.routes.analysis.AnalysisPipeline", StubPipeline)
    monkeypatch.setattr(
        "marshab.web.routes.analysis.get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(output_dir=tmp_path)),
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/analyze",
        json={
            "roi": [18.0, 18.6, 77.0, 77.8],
            "dataset": "mola_200m",
            "threshold": 0.7,
        },
    )

    assert response.status_code == 200
    assert StubPipeline.calls == ["mola_200m", "mola"]
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["top_site_id"] == 1
    assert payload["top_site_score"] == 0.91

