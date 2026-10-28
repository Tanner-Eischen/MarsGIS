"""DEM loading with explicit CRS, pixel-center coordinates, and masked elevations."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds, from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform

from marshab.exceptions import DataError
from marshab.models import BoundingBox


class DEMLoader:
    """Load north-up Mars DEMs; elevations and physical spacing are in meters.

    Geographic grids use the configured Mars radius and the center latitude to
    estimate east-west spacing. ``resolution_x_m`` and ``resolution_y_m`` retain
    anisotropy; ``resolution_m`` is their geometric mean for scalar consumers.
    """

    def __init__(self, mars_radius_eq: float = 3396190.0):
        self.mars_radius_eq = mars_radius_eq

    @staticmethod
    def _metadata(crs_value, transform_value) -> tuple[CRS, Affine]:
        if crs_value is None or transform_value is None:
            raise DataError("DEM requires an explicit CRS and affine transform")
        crs = CRS.from_user_input(crs_value)
        transform = Affine(*list(transform_value)[:6])
        if not all(np.isfinite(list(transform)[:6])):
            raise DataError("DEM affine transform must be finite")
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise DataError("Only north-up, unrotated DEM grids are supported")
        if not (crs.is_geographic or crs.is_projected):
            raise DataError("DEM CRS must be geographic or projected")
        return crs, transform

    def _array(self, elevation, crs: CRS, transform: Affine, attrs: dict) -> xr.DataArray:
        if elevation.ndim != 2 or min(elevation.shape) < 1:
            raise DataError("DEM must be a nonempty two-dimensional raster")
        height, width = elevation.shape
        x = transform.c + (np.arange(width) + 0.5) * transform.a
        y = transform.f + (np.arange(height) + 0.5) * transform.e
        xx, yy = np.meshgrid(x, y)
        if crs.is_geographic:
            lon, lat = xx, yy
            angular_units = crs.axis_info[0].unit_conversion_factor
            resolution_x = (
                abs(transform.a)
                * angular_units
                * self.mars_radius_eq
                * math.cos(math.radians(float(np.mean(y))))
            )
            resolution_y = abs(transform.e) * angular_units * self.mars_radius_eq
        else:
            geographic = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
            lon, lat = geographic.transform(xx.ravel().tolist(), yy.ravel().tolist())
            lon = np.asarray(lon).reshape(height, width)
            lat = np.asarray(lat).reshape(height, width)
            resolution_x = abs(transform.a) * crs.axis_info[0].unit_conversion_factor
            resolution_y = abs(transform.e) * crs.axis_info[1].unit_conversion_factor
        if (
            not np.isfinite([resolution_x, resolution_y]).all()
            or min(resolution_x, resolution_y) <= 0
        ):
            raise DataError("DEM physical pixel spacing must be finite and positive")
        return xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"x": x, "y": y, "lon": (("y", "x"), lon), "lat": (("y", "x"), lat)},
            attrs={
                **attrs,
                "crs": crs.to_wkt(),
                "transform": list(transform)[:6],
                "resolution_x_m": float(resolution_x),
                "resolution_y_m": float(resolution_y),
                "resolution_m": float(math.sqrt(resolution_x * resolution_y)),
                "units": "m",
            },
        )

    @staticmethod
    def _window(roi: BoundingBox, crs: CRS, transform: Affine, height: int, width: int) -> Window:
        bounds = (roi.lon_min, roi.lat_min, roi.lon_max, roi.lat_max)
        if crs.is_projected:
            projection = Transformer.from_crs(
                crs.geodetic_crs, crs, always_xy=True, force_over=True
            )
            bounds = projection.transform_bounds(*bounds, densify_pts=21)
        if not np.isfinite(bounds).all():
            raise DataError("ROI cannot be transformed into DEM CRS")
        left, bottom, right, top = array_bounds(height, width, transform)
        roi_left, roi_bottom, roi_right, roi_top = bounds
        if roi_left >= right or roi_right <= left or roi_bottom >= top or roi_top <= bottom:
            raise DataError("ROI does not overlap with DEM bounds")
        bounds = (
            max(left, roi_left),
            max(bottom, roi_bottom),
            min(right, roi_right),
            min(top, roi_top),
        )
        fractional = window_from_bounds(*bounds, transform=transform)
        # Include every intersecting pixel, without adding a pixel due to roundoff.
        col_start = max(0, math.floor(fractional.col_off + 1e-8))
        row_start = max(0, math.floor(fractional.row_off + 1e-8))
        col_stop = min(width, math.ceil(fractional.col_off + fractional.width - 1e-8))
        row_stop = min(height, math.ceil(fractional.row_off + fractional.height - 1e-8))
        if col_stop <= col_start or row_stop <= row_start:
            raise DataError("ROI does not contain a raster pixel")
        return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)

    def load(self, path: Path, roi: Optional[BoundingBox] = None) -> xr.DataArray:
        """Read a full DEM or an intersecting lon/lat ROI; masked cells become NaN.

        ROI bounds are interpreted in the source CRS's geographic reference
        system. Projected ROIs use a densified geographic-to-source envelope.
        Missing metadata or nonoverlapping ROIs fail rather than returning a
        different region or inferring georeferencing from filename/tags.
        """
        try:
            with rasterio.open(path) as src:
                crs, transform = self._metadata(src.crs, src.transform)
                window = self._window(roi, crs, transform, src.height, src.width) if roi else None
                elevation = (
                    src.read(1, window=window, masked=True).astype(np.float32).filled(np.nan)
                )
                if window is not None:
                    transform = window_transform(window, transform)
                return self._array(
                    elevation, crs, transform, {"nodata": src.nodata, "source_tags": src.tags()}
                )
        except DataError:
            raise
        except Exception as exc:
            raise DataError(
                "Failed to load DEM", details={"path": str(path), "error": str(exc)}
            ) from exc

    def clip_to_roi(self, dem: xr.DataArray, roi: BoundingBox) -> xr.DataArray:
        """Clip by intersecting pixels and update affine/coordinate metadata."""
        crs, transform = self._metadata(dem.attrs.get("crs"), dem.attrs.get("transform"))
        window = self._window(roi, crs, transform, *dem.shape)
        row, col = int(window.row_off), int(window.col_off)
        elevation = dem.values[row : row + int(window.height), col : col + int(window.width)]
        return self._array(elevation, crs, window_transform(window, transform), dem.attrs)

    def resample(
        self, dem: xr.DataArray, target_resolution_m: float, method: str = "bilinear"
    ) -> xr.DataArray:
        """Resample on the same bounds, retaining NaN masks and true pixel centers.

        Dimensions are rounded up to at least one cell. Exact target spacing is
        therefore approximate; output metadata records the resulting spacing.
        """
        if not np.isfinite(target_resolution_m) or target_resolution_m <= 0:
            raise DataError("Target resolution must be finite and positive")
        if method not in {"nearest", "bilinear", "cubic"}:
            raise DataError("Unsupported resampling method", details={"method": method})
        crs, transform = self._metadata(dem.attrs.get("crs"), dem.attrs.get("transform"))
        source = self._array(dem.values, crs, transform, dem.attrs)
        height = max(
            1, math.ceil(dem.shape[0] * source.attrs["resolution_y_m"] / target_resolution_m)
        )
        width = max(
            1, math.ceil(dem.shape[1] * source.attrs["resolution_x_m"] / target_resolution_m)
        )
        bounds = array_bounds(*dem.shape, transform)
        destination_transform = from_bounds(*bounds, width, height)
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=np.asarray(dem.values, dtype=np.float32),
            destination=destination,
            src_transform=transform,
            src_crs=crs.to_wkt(),
            src_nodata=np.nan,
            dst_transform=destination_transform,
            dst_crs=crs.to_wkt(),
            dst_nodata=np.nan,
            resampling=Resampling[method],
        )
        return self._array(destination, crs, destination_transform, dem.attrs)
