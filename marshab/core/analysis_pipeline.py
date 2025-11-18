"""Analysis pipeline for terrain analysis and site selection."""

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from skimage import measure

from marshab.config import get_config
from marshab.config.criteria_config import DEFAULT_CRITERIA
from marshab.core.data_manager import DataManager
from marshab.exceptions import AnalysisError
from marshab.processing.criteria import CriteriaExtractor
from marshab.processing.mcdm import MCDMEvaluator
from marshab.processing.terrain import TerrainAnalyzer
from marshab.types import BoundingBox, SiteCandidate, TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisResults:
    """Results from terrain analysis pipeline."""

    def __init__(
        self,
        sites: list[SiteCandidate],
        top_site_id: int,
        top_site_score: float,
        dem: Optional[xr.DataArray] = None,
        metrics: Optional[TerrainMetrics] = None,
        suitability: Optional[np.ndarray] = None,
        criteria: Optional[Dict] = None,
    ):
        """Initialize analysis results.

        Args:
            sites: List of identified site candidates
            top_site_id: ID of the top-ranked site
            top_site_score: Suitability score of the top site
            dem: DEM DataArray (optional, for navigation)
            metrics: TerrainMetrics (optional, for navigation)
            suitability: Suitability array (optional, for navigation)
            criteria: Criteria dictionary (optional, for navigation)
        """
        self.sites = sites
        self.top_site_id = top_site_id
        self.top_site_score = top_site_score
        self.dem = dem
        self.metrics = metrics
        self.suitability = suitability
        self.criteria = criteria

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
        criteria_weights: Optional[Dict[str, float]] = None,
        mcdm_method: Literal["weighted_sum", "topsis"] = "weighted_sum"
    ) -> AnalysisResults:
        """Run complete analysis pipeline.

        Args:
            roi: Region of interest
            dataset: DEM dataset to use
            threshold: Minimum suitability threshold
            criteria_weights: Optional custom weights (uses defaults if None)
            mcdm_method: MCDM method to use

        Returns:
            AnalysisResults with sites and suitability raster

        Raises:
            AnalysisError: If analysis fails
        """
        logger.info(
            "Starting analysis pipeline",
            roi=roi.model_dump(),
            dataset=dataset,
            threshold=threshold,
            mcdm_method=mcdm_method,
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
            
            # 2. Terrain analysis
            logger.info("Analyzing terrain")
            terrain_analyzer = TerrainAnalyzer(cell_size_m=cell_size_m)
            metrics = terrain_analyzer.analyze(dem)
            
            # 3. Extract criteria
            logger.info("Extracting criteria")
            extractor = CriteriaExtractor(dem, metrics)
            criteria = extractor.extract_all()
            
            # 4. Configure weights
            criteria_config = DEFAULT_CRITERIA
            
            if criteria_weights:
                # Update weights with custom values
                for name, weight in criteria_weights.items():
                    if name in criteria_config.criteria:
                        criteria_config.criteria[name].weight = weight
            
            criteria_config.validate_weights()
            
            weights = {name: c.weight for name, c in criteria_config.criteria.items()}
            beneficial = {name: c.beneficial for name, c in criteria_config.criteria.items()}
            
            # 5. MCDM evaluation
            logger.info("Evaluating suitability")
            suitability = MCDMEvaluator.evaluate(
                criteria,
                weights,
                beneficial,
                method=mcdm_method
            )
            
            # 6. Identify candidate sites
            logger.info("Identifying candidate sites")
            suitable_mask = suitability >= threshold
            
            # Safe logging
            try:
                mean_suit = float(np.nanmean(suitability)) if suitability.size > 0 else 0.0
                max_suit = float(np.nanmax(suitability)) if suitability.size > 0 else 0.0
                suitable_count = int(np.sum(suitable_mask)) if suitable_mask.size > 0 else 0
                total_count = int(suitable_mask.size) if suitable_mask.size > 0 else 0
                logger.info(
                    "MCDM evaluation complete",
                    mean_suitability=mean_suit,
                    max_suitability=max_suit,
                    suitable_pixels=suitable_count,
                    total_pixels=total_count
                )
            except Exception as e:
                logger.warning("Failed to log MCDM evaluation stats", error=str(e))
            
            # 7. Extract and rank sites
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
                region_elevation = metrics.elevation[region_mask]
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
                dem=dem,
                metrics=metrics,
                suitability=suitability,
                criteria=criteria,
            )

        except Exception as e:
            raise AnalysisError(
                "Analysis pipeline failed",
                details={"roi": roi.model_dump(), "error": str(e)},
            )
    
    def save_results(self, results: AnalysisResults, output_dir: Path) -> None:
        """Save analysis results for later use in navigation.
        
        Args:
            results: AnalysisResults object with all analysis data
            output_dir: Output directory
        """
        import pickle
        import json
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as pickle
        results_file = output_dir / "analysis_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'dem': results.dem,
                'metrics': results.metrics,
                'sites': results.sites,
                'suitability': results.suitability,
                'criteria': results.criteria
            }, f)
        
        logger.info(f"Saved analysis results pickle to {results_file}")
        
        # Save sites as GeoJSON
        if results.sites:
            sites_geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            for site in results.sites:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "site_id": site.site_id,
                        "area_km2": site.area_km2,
                        "mean_slope_deg": site.mean_slope_deg,
                        "mean_roughness": site.mean_roughness,
                        "mean_elevation_m": site.mean_elevation_m,
                        "suitability_score": site.suitability_score,
                        "rank": site.rank
                    }
                }
                
                if site.polygon_coords:
                    feature["geometry"] = {
                        "type": "Polygon",
                        "coordinates": [site.polygon_coords]
                    }
                else:
                    feature["geometry"] = {
                        "type": "Point",
                        "coordinates": [site.lon, site.lat]
                    }
                
                sites_geojson["features"].append(feature)
            
            sites_file = output_dir / "sites.geojson"
            with open(sites_file, 'w') as f:
                json.dump(sites_geojson, f, indent=2)
            
            logger.info(f"Saved sites GeoJSON to {sites_file}")
        
        # Save CSV file with sites data
        sites_csv_file = output_dir / "sites.csv"
        import pandas as pd
        sites_data = []
        for site in results.sites:
            sites_data.append({
                'site_id': site.site_id,
                'lat': site.lat,
                'lon': site.lon,
                'area_km2': site.area_km2,
                'mean_slope_deg': site.mean_slope_deg,
                'mean_roughness': site.mean_roughness,
                'mean_elevation_m': site.mean_elevation_m,
                'suitability_score': site.suitability_score,
                'rank': site.rank
            })
        df = pd.DataFrame(sites_data)
        df.to_csv(sites_csv_file, index=False)
        logger.info(f"Saved sites CSV to {sites_csv_file}")


