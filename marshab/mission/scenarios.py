"""High-level mission scenario orchestrators."""

from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from datetime import datetime

from marshab.core.analysis_pipeline import AnalysisPipeline, AnalysisResults
from marshab.core.navigation_engine import NavigationEngine
from marshab.types import BoundingBox, SiteCandidate
from marshab.config.preset_loader import PresetLoader
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LandingScenarioParams:
    """Parameters for landing site scenario."""
    roi: BoundingBox
    dataset: Literal["mola", "hirise", "ctx"] = "mola"
    preset_id: Optional[str] = None
    max_slope_deg: Optional[float] = None
    min_area_km2: Optional[float] = None
    suitability_threshold: float = 0.7
    custom_weights: Optional[Dict[str, float]] = None


@dataclass
class ScenarioLandingResult:
    """Results from landing site scenario."""
    scenario_id: str
    sites: List[SiteCandidate]
    top_site: Optional[SiteCandidate]
    metadata: Dict
    analysis_results: AnalysisResults
    created_at: datetime


@dataclass
class TraverseScenarioParams:
    """Parameters for rover traverse scenario."""
    start_site_id: int
    end_site_id: int
    analysis_dir: Path
    preset_id: Optional[str] = None
    rover_capabilities: Optional[Dict[str, float]] = None
    start_lat: Optional[float] = None
    start_lon: Optional[float] = None
    custom_weights: Optional[Dict[str, float]] = None


@dataclass
class RouteMetrics:
    """Route cost and risk metrics."""
    total_distance_m: float
    total_slope_cost: float
    total_roughness_cost: float
    total_shadow_cost: float
    estimated_energy_j: float
    risk_score: float
    num_waypoints: int


@dataclass
class ScenarioTraverseResult:
    """Results from rover traverse scenario."""
    route_id: str
    waypoints: List[Dict]
    route_metrics: RouteMetrics
    metadata: Dict
    created_at: datetime


def run_landing_site_scenario(params: LandingScenarioParams) -> ScenarioLandingResult:
    """Run landing site selection scenario.
    
    Orchestrates: DEM download → terrain analysis → MCDM evaluation → site extraction.
    
    Args:
        params: Landing scenario parameters
        
    Returns:
        ScenarioLandingResult with ranked sites and metadata
        
    Raises:
        ValueError: If parameters are invalid
    """
    logger.info("Starting landing site scenario", params=params)
    
    # Load preset if specified
    preset_loader = PresetLoader()
    preset = None
    criteria_weights = params.custom_weights
    
    if params.preset_id:
        preset = preset_loader.get_preset(params.preset_id, scope="site")
        if not preset:
            raise ValueError(f"Preset not found: {params.preset_id}")
        
        # Extract weights from preset
        preset_weights = preset.get_weights_dict()
        # Map preset weights to criteria weights format
        criteria_weights = {
            "slope": preset_weights.get("slope", 0.3),
            "roughness": preset_weights.get("roughness", 0.25),
            "elevation": preset_weights.get("elevation", 0.2),
            "solar_exposure": preset_weights.get("solar_exposure", 0.15),
            "resources": preset_weights.get("science_value", 0.1),  # Map science_value to resources
        }
        
        # Normalize weights
        total = sum(criteria_weights.values())
        if total > 0:
            criteria_weights = {k: v / total for k, v in criteria_weights.items()}
        
        # Apply preset thresholds if not overridden
        if params.max_slope_deg is None and preset.thresholds.max_slope_deg:
            params.max_slope_deg = preset.thresholds.max_slope_deg
        if params.min_area_km2 is None and preset.thresholds.min_site_area_km2:
            params.min_area_km2 = preset.thresholds.min_site_area_km2
    
    # Run analysis pipeline
    pipeline = AnalysisPipeline()
    
    # Prepare analysis parameters
    analysis_kwargs = {
        "roi": params.roi,
        "dataset": params.dataset,
        "threshold": params.suitability_threshold,
        "mcdm_method": "weighted_sum",
    }
    
    if criteria_weights:
        analysis_kwargs["criteria_weights"] = criteria_weights
    
    results = pipeline.run(**analysis_kwargs)
    
    # Filter sites by additional constraints if specified
    filtered_sites = results.sites
    if params.max_slope_deg:
        filtered_sites = [s for s in filtered_sites if s.mean_slope_deg <= params.max_slope_deg]
    if params.min_area_km2:
        filtered_sites = [s for s in filtered_sites if s.area_km2 >= params.min_area_km2]
    
    # Re-rank filtered sites
    filtered_sites.sort(key=lambda s: s.suitability_score, reverse=True)
    for i, site in enumerate(filtered_sites, 1):
        site.rank = i
    
    # Get top site
    top_site = filtered_sites[0] if filtered_sites else None
    
    # Generate scenario ID
    scenario_id = f"landing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare metadata
    metadata = {
        "roi": params.roi.model_dump(),
        "dataset": params.dataset,
        "preset_id": params.preset_id,
        "preset_name": preset.name if preset else None,
        "num_sites": len(filtered_sites),
        "threshold": params.suitability_threshold,
        "constraints": {
            "max_slope_deg": params.max_slope_deg,
            "min_area_km2": params.min_area_km2,
        }
    }
    
    logger.info(
        "Landing site scenario complete",
        scenario_id=scenario_id,
        num_sites=len(filtered_sites),
        top_site_id=top_site.site_id if top_site else None
    )
    
    return ScenarioLandingResult(
        scenario_id=scenario_id,
        sites=filtered_sites,
        top_site=top_site,
        metadata=metadata,
        analysis_results=results,
        created_at=datetime.now()
    )


def run_rover_traverse_scenario(params: TraverseScenarioParams) -> ScenarioTraverseResult:
    """Run rover traverse planning scenario.
    
    Orchestrates: Load sites → cost surface generation → A* pathfinding → waypoint creation.
    
    Args:
        params: Traverse scenario parameters
        
    Returns:
        ScenarioTraverseResult with route, waypoints, and metrics
        
    Raises:
        ValueError: If parameters are invalid
        NavigationError: If route planning fails
    """
    logger.info("Starting rover traverse scenario", params=params)
    
    # Load preset if specified
    preset_loader = PresetLoader()
    preset = None
    route_weights = params.custom_weights
    
    if params.preset_id:
        preset = preset_loader.get_preset(params.preset_id, scope="route")
        if not preset:
            raise ValueError(f"Route preset not found: {params.preset_id}")
        
        # Extract route weights from preset
        preset_weights = preset.get_weights_dict()
        route_weights = {
            "distance": preset_weights.get("distance", 0.3),
            "slope_penalty": preset_weights.get("slope_penalty", 0.25),
            "roughness_penalty": preset_weights.get("roughness_penalty", 0.2),
            "elevation_penalty": preset_weights.get("elevation_penalty", 0.25),
        }
        
        # Normalize weights
        total = sum(route_weights.values())
        if total > 0:
            route_weights = {k: v / total for k, v in route_weights.items()}
    
    # Load sites from analysis results
    import pandas as pd
    sites_csv = params.analysis_dir / "sites.csv"
    if not sites_csv.exists():
        raise ValueError(f"Sites file not found: {sites_csv}")
    
    sites_df = pd.read_csv(sites_csv)
    
    # Find start and end sites
    start_site = sites_df[sites_df["site_id"] == params.start_site_id]
    end_site = sites_df[sites_df["site_id"] == params.end_site_id]
    
    if start_site.empty:
        raise ValueError(f"Start site {params.start_site_id} not found")
    if end_site.empty:
        raise ValueError(f"End site {params.end_site_id} not found")
    
    start_lat = params.start_lat if params.start_lat else float(start_site["lat"].iloc[0])
    start_lon = params.start_lon if params.start_lon else float(start_site["lon"].iloc[0])
    end_lat = float(end_site["lat"].iloc[0])
    end_lon = float(end_site["lon"].iloc[0])
    
    # Plan route using navigation engine
    engine = NavigationEngine()
    
    # Use end site as target
    waypoints_df = engine.plan_to_site(
        site_id=params.end_site_id,
        analysis_dir=params.analysis_dir,
        start_lat=start_lat,
        start_lon=start_lon,
        max_waypoint_spacing_m=100.0,
        max_slope_deg=params.rover_capabilities.get("max_slope_deg", 25.0) if params.rover_capabilities else 25.0
    )
    
    # Calculate route metrics
    if len(waypoints_df) > 0:
        # Calculate total distance
        total_distance = 0.0
        for i in range(len(waypoints_df) - 1):
            x1 = waypoints_df.iloc[i]["x_meters"]
            y1 = waypoints_df.iloc[i]["y_meters"]
            x2 = waypoints_df.iloc[i + 1]["x_meters"]
            y2 = waypoints_df.iloc[i + 1]["y_meters"]
            segment_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            total_distance += segment_dist
        
        # Estimate energy (simplified: distance * base_energy + slope_penalty)
        estimated_energy = total_distance * 100.0  # 100 J/m base
        
        # Calculate risk score (0-1, higher = riskier)
        # Simplified: based on distance and number of waypoints
        risk_score = min(1.0, total_distance / 10000.0)  # Normalize to 10km
        
    else:
        total_distance = 0.0
        estimated_energy = 0.0
        risk_score = 0.0
    
    route_metrics = RouteMetrics(
        total_distance_m=total_distance,
        total_slope_cost=0.0,  # Would be calculated from actual route
        total_roughness_cost=0.0,
        total_shadow_cost=0.0,
        estimated_energy_j=estimated_energy,
        risk_score=risk_score,
        num_waypoints=len(waypoints_df)
    )
    
    # Convert waypoints to dict format
    waypoints = waypoints_df.to_dict('records')
    
    # Generate route ID
    route_id = f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare metadata
    metadata = {
        "start_site_id": params.start_site_id,
        "end_site_id": params.end_site_id,
        "preset_id": params.preset_id,
        "preset_name": preset.name if preset else None,
        "rover_capabilities": params.rover_capabilities,
    }
    
    logger.info(
        "Rover traverse scenario complete",
        route_id=route_id,
        num_waypoints=len(waypoints),
        total_distance_m=total_distance
    )
    
    return ScenarioTraverseResult(
        route_id=route_id,
        waypoints=waypoints,
        route_metrics=route_metrics,
        metadata=metadata,
        created_at=datetime.now()
    )

