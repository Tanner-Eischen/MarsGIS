"""Data product export functionality."""

from pathlib import Path
from typing import Dict, Optional, Union
import json
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import xarray as xr

from marshab.core.analysis_pipeline import AnalysisResults
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


def export_suitability_geotiff(
    roi: BoundingBox,
    dataset: str,
    weights: Dict[str, float],
    output_path: Path,
    dem: Optional[xr.DataArray] = None,
    suitability: Optional[np.ndarray] = None
) -> Path:
    """Export suitability scores as GeoTIFF.
    
    Args:
        roi: Region of interest
        dataset: Dataset name
        weights: Criteria weights used
        output_path: Output file path
        dem: Optional DEM DataArray (if already loaded)
        suitability: Optional suitability array (if already calculated)
        
    Returns:
        Path to exported GeoTIFF file
        
    Raises:
        ValueError: If DEM or suitability data not available
    """
    logger.info("Exporting suitability GeoTIFF", output_path=str(output_path))
    
    if suitability is None or dem is None:
        raise ValueError("DEM and suitability data must be provided or calculated")
    
    # Get DEM bounds and transform
    if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
        bounds = dem.rio.bounds()
        transform = dem.rio.transform()
    else:
        # Create transform from ROI bounds
        height, width = suitability.shape
        transform = from_bounds(
            roi.lon_min,
            roi.lat_min,
            roi.lon_max,
            roi.lat_max,
            width,
            height
        )
        bounds = (roi.lon_min, roi.lat_min, roi.lon_max, roi.lat_max)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write GeoTIFF
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=suitability.shape[0],
        width=suitability.shape[1],
        count=1,
        dtype=suitability.dtype,
        crs='EPSG:4326',  # WGS84 - would need Mars CRS in production
        transform=transform,
        compress='lzw',
        nodata=-9999.0
    ) as dst:
        # Replace NaN with nodata
        suitability_clean = np.where(
            np.isfinite(suitability),
            suitability,
            -9999.0
        )
        dst.write(suitability_clean, 1)
        
        # Add metadata
        dst.update_tags(
            title="Mars Landing Site Suitability",
            dataset=dataset,
            roi=f"{roi.lat_min},{roi.lat_max},{roi.lon_min},{roi.lon_max}",
            weights=json.dumps(weights),
            created=datetime.now().isoformat()
        )
    
    logger.info("Suitability GeoTIFF exported", file_size_mb=output_path.stat().st_size / 1e6)
    return output_path


def generate_analysis_report(
    project_or_run: Union[Dict, AnalysisResults],
    output_path: Optional[Path] = None,
    format: str = "markdown"
) -> Path:
    """Generate analysis report in Markdown or HTML format.
    
    Args:
        project_or_run: Project dictionary or AnalysisResults object
        output_path: Optional output file path (auto-generated if None)
        format: Output format ("markdown" or "html")
        
    Returns:
        Path to generated report file
    """
    logger.info("Generating analysis report", format=format)
    
    # Extract data from project_or_run
    if isinstance(project_or_run, AnalysisResults):
        sites = project_or_run.sites
        top_site = project_or_run.sites[0] if project_or_run.sites else None
        metadata = {
            "num_sites": len(sites),
            "top_site_id": project_or_run.top_site_id,
            "top_site_score": project_or_run.top_site_score,
        }
    else:
        sites = project_or_run.get("sites", [])
        top_site = project_or_run.get("top_site")
        metadata = project_or_run.get("metadata", {})
    
    # Generate report content
    if format == "markdown":
        content = _generate_markdown_report(sites, top_site, metadata)
        ext = ".md"
    elif format == "html":
        content = _generate_html_report(sites, top_site, metadata)
        ext = ".html"
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Determine output path
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("data/output") / f"analysis_report_{timestamp}{ext}"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info("Analysis report generated", path=str(output_path))
    return output_path


def _generate_markdown_report(
    sites: list,
    top_site: Optional[Dict],
    metadata: Dict
) -> str:
    """Generate Markdown report content."""
    lines = [
        "# Mars Landing Site Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- **Total Sites Found:** {len(sites)}",
        f"- **Top Site ID:** {metadata.get('top_site_id', 'N/A')}",
        f"- **Top Site Score:** {metadata.get('top_site_score', 0.0):.3f}",
        "",
    ]
    
    if top_site:
        lines.extend([
            "## Top Site Details",
            "",
            f"- **Site ID:** {top_site.site_id if hasattr(top_site, 'site_id') else top_site.get('site_id')}",
            f"- **Rank:** {top_site.rank if hasattr(top_site, 'rank') else top_site.get('rank')}",
            f"- **Suitability Score:** {top_site.suitability_score if hasattr(top_site, 'suitability_score') else top_site.get('suitability_score', 0.0):.3f}",
            f"- **Area:** {top_site.area_km2 if hasattr(top_site, 'area_km2') else top_site.get('area_km2', 0.0):.2f} km²",
            f"- **Location:** {top_site.lat if hasattr(top_site, 'lat') else top_site.get('lat', 0.0):.4f}°N, {top_site.lon if hasattr(top_site, 'lon') else top_site.get('lon', 0.0):.4f}°E",
            f"- **Mean Slope:** {top_site.mean_slope_deg if hasattr(top_site, 'mean_slope_deg') else top_site.get('mean_slope_deg', 0.0):.2f}°",
            "",
        ])
    
    if len(sites) > 0:
        lines.extend([
            "## Top 10 Sites",
            "",
            "| Rank | Site ID | Score | Area (km²) | Slope (°) | Location |",
            "|------|---------|-------|------------|-----------|----------|",
        ])
        
        for site in sites[:10]:
            site_id = site.site_id if hasattr(site, 'site_id') else site.get('site_id')
            rank = site.rank if hasattr(site, 'rank') else site.get('rank')
            score = site.suitability_score if hasattr(site, 'suitability_score') else site.get('suitability_score', 0.0)
            area = site.area_km2 if hasattr(site, 'area_km2') else site.get('area_km2', 0.0)
            slope = site.mean_slope_deg if hasattr(site, 'mean_slope_deg') else site.get('mean_slope_deg', 0.0)
            lat = site.lat if hasattr(site, 'lat') else site.get('lat', 0.0)
            lon = site.lon if hasattr(site, 'lon') else site.get('lon', 0.0)
            
            lines.append(
                f"| {rank} | {site_id} | {score:.3f} | {area:.2f} | {slope:.2f} | {lat:.2f}°N, {lon:.2f}°E |"
            )
    
    return "\n".join(lines)


def _generate_html_report(
    sites: list,
    top_site: Optional[Dict],
    metadata: Dict
) -> str:
    """Generate HTML report content."""
    markdown = _generate_markdown_report(sites, top_site, metadata)
    
    # Simple Markdown to HTML conversion (basic)
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mars Landing Site Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
{markdown.replace('|', '</td><td>').replace('\n|', '<tr><td>').replace('|', '</td></tr>')}
</body>
</html>"""
    
    return html

