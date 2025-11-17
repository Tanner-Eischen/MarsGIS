"""Analysis pipeline for terrain analysis and site selection."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

from marshab.config import get_config
from marshab.core.data_manager import DataManager
from marshab.exceptions import AnalysisError
from marshab.processing.mcdm import MCDMEvaluator
from marshab.processing.terrain import TerrainAnalyzer
from marshab.types import BoundingBox, SiteCandidate
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisResults:
    """Results from terrain analysis pipeline."""

    def __init__(
        self,
        sites: list[SiteCandidate],
        top_site_id: int,
        top_site_score: float,
    ):
        """Initialize analysis results.

        Args:
            sites: List of identified site candidates
            top_site_id: ID of the top-ranked site
            top_site_score: Suitability score of the top site
        """
        self.sites = sites
        self.top_site_id = top_site_id
        self.top_site_score = top_site_score

    def save(self, output_dir: Path) -> None:
        """Save analysis results to output directory.

        Args:
            output_dir: Directory to save results to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save sites as CSV
        sites_data = []
        for site in self.sites:
            site_dict = site.model_dump()
            # Convert polygon_coords list to string for CSV storage
            if site_dict.get('polygon_coords'):
                import json
                site_dict['polygon_coords'] = json.dumps(site_dict['polygon_coords'])
            sites_data.append(site_dict)
        sites_df = pd.DataFrame(sites_data)
        sites_df.to_csv(output_dir / "sites.csv", index=False)

        logger.info("Saved analysis results", output_dir=str(output_dir), num_sites=len(self.sites))


class AnalysisPipeline:
    """Orchestrates geospatial analysis workflow."""

    def __init__(self):
        """Initialize analysis pipeline."""
        self.data_manager = DataManager()
        logger.info("Initialized AnalysisPipeline")

    def run(
        self,
        roi: BoundingBox,
        dataset: Literal["mola", "hirise", "ctx"] = "mola",
        threshold: float = 0.7,
    ) -> AnalysisResults:
        """Run complete terrain analysis pipeline.

        Args:
            roi: Region of interest
            dataset: Dataset to use
            threshold: Suitability threshold

        Returns:
            AnalysisResults with identified sites

        Raises:
            AnalysisError: If analysis fails
        """
        logger.info(
            "Starting analysis pipeline",
            roi=roi.model_dump(),
            dataset=dataset,
            threshold=threshold,
        )

        try:
            config = get_config()
            
            # 1. Load DEM for ROI
            logger.info("Loading DEM data", dataset=dataset, roi=roi.model_dump())
            dem = self.data_manager.get_dem_for_roi(roi, dataset, download=True, clip=True)
            
            # Extract cell size from DEM (use resolution from config if available, else estimate from bounds)
            if dataset in config.data_sources:
                cell_size_m = config.data_sources[dataset].resolution_m
            else:
                # Estimate from DEM bounds
                if hasattr(dem, 'rio') and hasattr(dem.rio, 'res'):
                    cell_size_m = float(abs(dem.rio.res[0]))
                else:
                    # Fallback: estimate from bounds
                    bounds = dem.rio.bounds()
                    width_m = (bounds.right - bounds.left) * 111000 * np.cos(np.radians(roi.lat_min))
                    cell_size_m = width_m / dem.shape[1] if dem.shape[1] > 0 else 200.0
            
            logger.info("DEM loaded", shape=dem.shape, cell_size_m=cell_size_m)
            
            # 2. Calculate terrain metrics
            logger.info("Calculating terrain metrics")
            terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
            metrics = terrain_analyzer.analyze(dem)
            
            elevation = dem.values.astype(np.float32)
            
            # 3. Apply MCDM evaluation
            logger.info("Applying MCDM evaluation", threshold=threshold)
            criteria_weights = config.analysis.criteria_weights.normalize()
            
            # Prepare criteria dict - use available metrics
            criteria = {
                "slope": metrics.slope,
                "roughness": metrics.roughness,
                "elevation": elevation,  # Use elevation as proxy for elevation criterion
            }
            
            # Define beneficial dict (lower is better for slope/roughness, higher for elevation)
            beneficial = {
                "slope": False,  # Lower slope is better
                "roughness": False,  # Lower roughness is better
                "elevation": True,  # Higher elevation can be better (but this is simplified)
            }
            
            # Get weights dict (only include criteria we have)
            weights_dict = {}
            for key in criteria.keys():
                if hasattr(criteria_weights, key):
                    weights_dict[key] = getattr(criteria_weights, key)
            
            # Normalize weights to sum to 1.0 for available criteria
            total_weight = sum(weights_dict.values())
            if total_weight > 0:
                weights_dict = {k: v / total_weight for k, v in weights_dict.items()}
            else:
                # Default equal weights if none specified
                weights_dict = {k: 1.0 / len(criteria) for k in criteria.keys()}
            
            # Calculate suitability scores
            suitability = MCDMEvaluator.weighted_sum(criteria, weights_dict, beneficial)
            
            # Apply threshold
            suitable_mask = suitability >= threshold
            
            # Safe logging
            try:
                mean_suit = float(np.nanmean(suitability)) if suitability.size > 0 else 0.0
                suitable_count = int(np.sum(suitable_mask)) if suitable_mask.size > 0 else 0
                total_count = int(suitable_mask.size) if suitable_mask.size > 0 else 0
                logger.info(
                    "MCDM evaluation complete",
                    mean_suitability=mean_suit,
                    suitable_pixels=suitable_count,
                    total_pixels=total_count
                )
            except Exception as e:
                logger.warning("Failed to log MCDM evaluation stats", error=str(e))
            
            # 4. Extract and rank sites
            logger.info("Extracting candidate sites")
            
            # Find connected regions above threshold
            try:
                labeled_array, num_features = ndimage.label(suitable_mask)
            except Exception as e:
                logger.error("Failed to label suitable regions", error=str(e))
                return AnalysisResults(
                    sites=[],
                    top_site_id=0,
                    top_site_score=0.0,
                )
            
            if num_features == 0:
                logger.warning("No suitable sites found above threshold", threshold=threshold)
                return AnalysisResults(
                    sites=[],
                    top_site_id=0,
                    top_site_score=0.0,
                )
            
            # Calculate properties for each region
            sites: list[SiteCandidate] = []
            cell_area_km2 = (cell_size_m ** 2) / 1e6  # Convert m² to km²
            
            for site_id in range(1, num_features + 1):
                region_mask = labeled_array == site_id
                region_size = np.sum(region_mask)
                
                # Skip empty regions
                if region_size == 0:
                    continue
                
                area_km2 = region_size * cell_area_km2
                
                # Check minimum area threshold
                if area_km2 < config.analysis.min_site_area_km2:
                    continue
                
                # Extract region data
                region_slope = metrics.slope[region_mask]
                region_roughness = metrics.roughness[region_mask]
                region_elevation = elevation[region_mask]
                region_suitability = suitability[region_mask]
                
                # Calculate mean properties for this region (handle empty arrays)
                mean_slope = float(np.nanmean(region_slope)) if region_slope.size > 0 else 0.0
                mean_roughness = float(np.nanmean(region_roughness)) if region_roughness.size > 0 else 0.0
                mean_elevation = float(np.nanmean(region_elevation)) if region_elevation.size > 0 else 0.0
                mean_suitability = float(np.nanmean(region_suitability)) if region_suitability.size > 0 else 0.0
                
                # Find centroid (in pixel coordinates)
                try:
                    centroid_y, centroid_x = ndimage.center_of_mass(region_mask)
                    # Ensure centroid is valid
                    if not (np.isfinite(centroid_y) and np.isfinite(centroid_x)):
                        # Use geometric center as fallback
                        y_indices, x_indices = np.where(region_mask)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            centroid_y = float(np.mean(y_indices))
                            centroid_x = float(np.mean(x_indices))
                        else:
                            # Skip this site if we can't find a centroid
                            continue
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Failed to calculate centroid for site {site_id}, skipping", error=str(e))
                    continue
                
                # Convert pixel coordinates to lat/lon
                try:
                    if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
                        bounds = dem.rio.bounds()
                        lon = float(bounds.left + (centroid_x / dem.shape[1]) * (bounds.right - bounds.left))
                        lat = float(bounds.top - (centroid_y / dem.shape[0]) * (bounds.top - bounds.bottom))
                    else:
                        # Fallback: estimate from ROI bounds
                        lon = float(roi.lon_min + (centroid_x / dem.shape[1]) * (roi.lon_max - roi.lon_min))
                        lat = float(roi.lat_max - (centroid_y / dem.shape[0]) * (roi.lat_max - roi.lat_min))
                except (ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Failed to convert centroid to lat/lon for site {site_id}, skipping", error=str(e))
                    continue
                
                # Extract polygon boundary from region_mask
                polygon_coords = None
                try:
                    # Find contours of the region
                    contours = measure.find_contours(region_mask.astype(float), 0.5)
                    if len(contours) > 0:
                        # Use the largest contour (outer boundary)
                        largest_contour = max(contours, key=len)
                        # Convert pixel coordinates to lat/lon
                        if hasattr(dem, 'rio') and hasattr(dem.rio, 'bounds'):
                            bounds = dem.rio.bounds()
                            lon_min = bounds.left
                            lon_max = bounds.right
                            lat_max = bounds.top
                            lat_min = bounds.bottom
                        else:
                            lon_min = roi.lon_min
                            lon_max = roi.lon_max
                            lat_max = roi.lat_max
                            lat_min = roi.lat_min
                        
                        # Convert contour coordinates (y, x) to (lon, lat)
                        polygon_coords = []
                        for y_px, x_px in largest_contour:
                            lon_px = float(lon_min + (x_px / dem.shape[1]) * (lon_max - lon_min))
                            lat_px = float(lat_max - (y_px / dem.shape[0]) * (lat_max - lat_min))
                            polygon_coords.append([lon_px, lat_px])
                        
                        # Ensure polygon is closed (first point == last point)
                        if len(polygon_coords) > 0 and polygon_coords[0] != polygon_coords[-1]:
                            polygon_coords.append(polygon_coords[0])
                except Exception as e:
                    logger.warning(f"Failed to extract polygon for site {site_id}, using point geometry", error=str(e))
                
                # Create site with temporary rank (will be updated after sorting)
                site = SiteCandidate(
                    site_id=site_id,
                    geometry_type="POLYGON" if polygon_coords else "POINT",
                    area_km2=area_km2,
                    lat=lat,
                    lon=lon,
                    mean_slope_deg=mean_slope,
                    mean_roughness=mean_roughness,
                    mean_elevation_m=mean_elevation,
                    suitability_score=mean_suitability,
                    rank=len(sites) + 1,  # Temporary rank, will be updated after sorting
                    polygon_coords=polygon_coords,
                )
                sites.append(site)
            
            # Sort by suitability score (descending) and assign ranks
            sites.sort(key=lambda s: s.suitability_score, reverse=True)
            for rank, site in enumerate(sites, start=1):
                # Update rank by creating a new instance (Pydantic models are immutable)
                site_dict = site.model_dump()
                site_dict['rank'] = rank
                sites[rank - 1] = SiteCandidate(**site_dict)
            
            # Get top site
            top_site_id = sites[0].site_id if sites else 0
            top_site_score = sites[0].suitability_score if sites else 0.0
            
            logger.info(
                "Site extraction complete",
                num_sites=len(sites),
                top_site_id=top_site_id,
                top_site_score=top_site_score,
            )

            return AnalysisResults(
                sites=sites,
                top_site_id=top_site_id,
                top_site_score=top_site_score,
            )

        except Exception as e:
            raise AnalysisError(
                "Analysis pipeline failed",
                details={"roi": roi.model_dump(), "error": str(e)},
            )


