"""Example plugin demonstrating plugin interface."""

from typing import Any

import numpy as np
import xarray as xr

from marshab.plugins.base import CriterionPlugin


class ExampleCriterionPlugin(CriterionPlugin):
    """Example criterion plugin that calculates a custom metric."""

    def get_criteria(self) -> list[dict[str, Any]]:
        """Return example criterion definition."""
        return [{
            "id": "example_metric",
            "name": "Example Metric",
            "description": "An example custom criterion for demonstration",
            "beneficial": True,
            "unit": "normalized"
        }]

    def calculate(self, dem: xr.DataArray, metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Calculate example metric (simplified: based on elevation)."""
        elevation = dem.values

        # Example: normalize elevation to 0-1 range
        if elevation.size == 0:
            return np.array([])

        valid_mask = np.isfinite(elevation)
        if not np.any(valid_mask):
            return np.zeros_like(elevation)

        valid_elevation = elevation[valid_mask]
        elev_min = np.nanmin(valid_elevation)
        elev_max = np.nanmax(valid_elevation)

        if elev_max == elev_min:
            return np.ones_like(elevation) * 0.5

        normalized = (elevation - elev_min) / (elev_max - elev_min)
        normalized[~valid_mask] = 0.0

        return normalized

