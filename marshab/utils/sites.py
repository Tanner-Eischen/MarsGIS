"""Shared helpers for loading site data from analysis outputs."""

from __future__ import annotations

from pathlib import Path


def load_site_coords(
    analysis_dir: Path | str,
    site_id: int,
    include_elevation: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """Load lat/lon (and optionally elevation) for a site from sites.csv.

    Args:
        analysis_dir: Path to analysis results directory containing sites.csv
        site_id: Site identifier
        include_elevation: If True, return (lat, lon, elevation)

    Returns:
        (lat, lon) or (lat, lon, mean_elevation_m)

    Raises:
        FileNotFoundError: If sites.csv does not exist
        ValueError: If site_id not found
    """
    path = Path(analysis_dir)
    sites_csv = path / "sites.csv"
    if not sites_csv.exists():
        raise FileNotFoundError(f"Sites file not found: {sites_csv}")

    import pandas as pd

    df = pd.read_csv(sites_csv)
    row = df[df["site_id"] == site_id]
    if row.empty:
        raise ValueError(f"Site {site_id} not found in {sites_csv}")

    lat = float(row["lat"].iloc[0])
    lon = float(row["lon"].iloc[0])
    if include_elevation:
        elev = float(row["mean_elevation_m"].iloc[0]) if "mean_elevation_m" in row.columns else 0.0
        return (lat, lon, elev)
    return (lat, lon)
