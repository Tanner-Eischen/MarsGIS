"""Visibility and Viewshed analysis."""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy import ndimage

from marshab.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class VisibilityResult:
    viewshed_map: np.ndarray  # Boolean or Probability mask
    visible_area_km2: float
    max_range_m: float

class VisibilityAnalyzer:
    def __init__(self, cell_size_m: float = 200.0):
        self.cell_size_m = cell_size_m

    def calculate_viewshed(
        self,
        dem: xr.DataArray,
        observer_pos: tuple[int, int],  # (row, col)
        observer_height_m: float = 2.0,
        target_height_m: float = 0.0,
        max_radius_px: int = 100
    ) -> VisibilityResult:
        """Calculate Line-of-Sight (LOS) viewshed from observer position.

        Uses a simplified R2 (Reference Plane) algorithm or Bresenham-like
        sampling for speed.
        """
        elevation = dem.values.astype(np.float32)
        rows, cols = elevation.shape
        obs_r, obs_c = observer_pos

        # Initialize viewshed map (0=hidden, 1=visible)
        viewshed = np.zeros_like(elevation, dtype=bool)

        # Mark observer as visible
        if 0 <= obs_r < rows and 0 <= obs_c < cols:
            viewshed[obs_r, obs_c] = True
            obs_elev = elevation[obs_r, obs_c] + observer_height_m
        else:
            return VisibilityResult(viewshed, 0.0, 0.0)

        # Sampling rays (360 degrees)
        # For performance, we just cast rays to edge of max_radius
        # A full Bresenham for every pixel is O(N^3), too slow for Python without numba.
        # We'll use a simplified localized horizon check for 'Real Value' demo speed.

        # Iterate over concentric rings? Or just sparse sampling?
        # Let's do 72 rays (every 5 degrees)
        for angle in np.linspace(0, 2*np.pi, 72):
            dr = np.sin(angle)
            dc = np.cos(angle)

            max_slope = -9999.0

            # March along ray
            for r in range(1, max_radius_px):
                curr_r = int(obs_r + r * dr)
                curr_c = int(obs_c + r * dc)

                if not (0 <= curr_r < rows and 0 <= curr_c < cols):
                    break

                # Calculate slope to this point
                dist_m = r * self.cell_size_m
                target_elev = elevation[curr_r, curr_c] + target_height_m

                # Angle of elevation from observer to target
                slope = (target_elev - obs_elev) / dist_m

                if slope >= max_slope:
                    viewshed[curr_r, curr_c] = True
                    max_slope = slope
                else:
                    # Hidden
                    viewshed[curr_r, curr_c] = False

        # Fill gaps (simple dilation to make it look like a solid region)
        viewshed = ndimage.binary_dilation(viewshed, iterations=1)

        visible_pixels = np.sum(viewshed)
        area_km2 = (visible_pixels * (self.cell_size_m**2)) / 1e6

        return VisibilityResult(
            viewshed_map=viewshed,
            visible_area_km2=float(area_km2),
            max_range_m=float(max_radius_px * self.cell_size_m)
        )

