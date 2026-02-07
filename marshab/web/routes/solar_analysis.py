"""Solar analysis API endpoints."""


import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.analysis.solar_potential import MissionImpacts, SolarPotentialAnalyzer
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/solar", tags=["solar"])


@router.get("/health")
async def solar_health():
    """Health check for solar analysis endpoint."""
    return {"status": "ok", "message": "Solar analysis endpoint is available"}


class SolarAnalysisRequest(BaseModel):
    """Request for solar analysis."""
    roi: dict[str, float] = Field(
        ...,
        description="Bounding box: lat_min, lat_max, lon_min, lon_max"
    )
    dataset: str = Field("mola", description="DEM dataset")
    sun_azimuth: float = Field(0.0, ge=0, le=360, description="Sun azimuth angle in degrees (0=North)")
    sun_altitude: float = Field(45.0, ge=0, le=90, description="Sun altitude angle in degrees (0=horizon)")
    panel_efficiency: float = Field(0.25, ge=0, le=1, description="Solar panel efficiency (0-1)")
    panel_area_m2: float = Field(100.0, ge=0, description="Total panel area in square meters")
    battery_capacity_kwh: float = Field(50.0, ge=0, description="Baseline battery capacity in kWh")
    daily_power_needs_kwh: float = Field(20.0, ge=0, description="Daily power requirements in kWh")
    battery_cost_per_kwh: float = Field(1000.0, ge=0, description="Cost per kWh of battery storage")
    mission_duration_days: float = Field(500.0, ge=0, description="Baseline mission duration in days")


class SolarStatistics(BaseModel):
    """Solar potential statistics."""
    min: float
    max: float
    mean: float
    std: float
    min_irradiance_w_per_m2: float
    max_irradiance_w_per_m2: float
    mean_irradiance_w_per_m2: float


class MissionImpactsResponse(BaseModel):
    """Mission impact calculations."""
    power_generation_kwh_per_day: float
    power_surplus_kwh_per_day: float
    mission_duration_extension_days: float
    cost_savings_usd: float
    battery_reduction_kwh: float


class SolarAnalysisResponse(BaseModel):
    """Response from solar analysis."""
    solar_potential_map: list[list[float]] = Field(..., description="Solar potential map (normalized 0-1)")
    irradiance_map: list[list[float]] = Field(..., description="Irradiance map (W/m²)")
    statistics: SolarStatistics
    mission_impacts: MissionImpactsResponse
    shape: dict[str, int] = Field(..., description="Map dimensions (rows, cols)")


@router.post("/analyze", response_model=SolarAnalysisResponse)
async def analyze_solar_potential(request: SolarAnalysisRequest):
    """Analyze solar potential for a region.

    Returns solar potential map, statistics, and mission impact calculations.
    """
    try:
        # Run analysis pipeline to get terrain metrics
        pipeline = AnalysisPipeline()
        roi = BoundingBox(**request.roi)

        logger.info("Running analysis pipeline for solar analysis", roi=roi.model_dump())
        use_synthetic = False
        try:
            results = pipeline.run(
                roi,
                dataset=request.dataset,
                threshold=0.5,
                criteria_weights=None
            )
            if results.dem is None or results.metrics is None:
                use_synthetic = True
        except Exception as e:
            logger.warning("Analysis pipeline failed for solar, using synthetic fallback", error=str(e))
            use_synthetic = True

        # Get cell size from DEM (or synthetic default)
        if not use_synthetic:
            if hasattr(results.dem, 'rio') and hasattr(results.dem.rio, 'res'):
                cell_size_m = float(abs(results.dem.rio.res[0]))
            elif hasattr(results.dem, 'rio') and hasattr(results.dem.rio, 'bounds'):
                try:
                    bounds = results.dem.rio.bounds()
                    width_m = (bounds.right - bounds.left) * 111000 * np.cos(np.radians(roi.lat_min))
                    cell_size_m = width_m / results.dem.shape[1] if results.dem.shape[1] > 0 else 200.0
                except (AttributeError, TypeError):
                    width_m = (roi.lon_max - roi.lon_min) * 111000 * np.cos(np.radians(roi.lat_min))
                    cell_size_m = width_m / results.dem.shape[1] if results.dem.shape[1] > 0 else 200.0
            else:
                width_m = (roi.lon_max - roi.lon_min) * 111000 * np.cos(np.radians(roi.lat_min))
                cell_size_m = width_m / results.dem.shape[1] if results.dem.shape[1] > 0 else 200.0
        else:
            cell_size_m = 200.0

        # Analyze solar potential
        analyzer = SolarPotentialAnalyzer(cell_size_m=cell_size_m)
        if not use_synthetic:
            solar_result = analyzer.calculate_solar_potential(
                results.dem,
                results.metrics,
                sun_azimuth=request.sun_azimuth,
                sun_altitude=request.sun_altitude
            )
        else:
            # Synthetic solar potential and irradiance
            rows, cols = 300, 400
            y = np.linspace(0, 1, rows)
            x = np.linspace(0, 1, cols)
            xx, yy = np.meshgrid(x, y)
            solar_potential = (np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) + 1.0) / 2.0
            irradiance = solar_potential * 590.0  # Approx Mars solar constant scaling
            stats = {
                "min": float(np.nanmin(solar_potential)),
                "max": float(np.nanmax(solar_potential)),
                "mean": float(np.nanmean(solar_potential)),
                "std": float(np.nanstd(solar_potential)),
                "min_irradiance_w_per_m2": float(np.nanmin(irradiance)),
                "max_irradiance_w_per_m2": float(np.nanmax(irradiance)),
                "mean_irradiance_w_per_m2": float(np.nanmean(irradiance)),
            }
            class SynthResult:
                def __init__(self, pot, irr, statistics):
                    self.solar_potential_map = pot
                    self.irradiance_map = irr
                    self.statistics = statistics
            solar_result = SynthResult(solar_potential, irradiance, stats)

        # Calculate mission impacts
        if not use_synthetic:
            mission_impacts = analyzer.calculate_mission_impacts(
                solar_result,
                panel_efficiency=request.panel_efficiency,
                panel_area_m2=request.panel_area_m2,
                battery_capacity_kwh=request.battery_capacity_kwh,
                daily_power_needs_kwh=request.daily_power_needs_kwh,
                battery_cost_per_kwh=request.battery_cost_per_kwh,
                mission_duration_days=request.mission_duration_days
            )
        else:
            # Simple synthetic mission impact calculation
            avg_irr = float(np.nanmean(solar_result.irradiance_map))
            hours = 6.0
            generation = avg_irr * request.panel_efficiency * request.panel_area_m2 * hours / 1000.0
            surplus = generation - request.daily_power_needs_kwh
            battery_reduction = float(max(0.0, min(request.daily_power_needs_kwh, surplus)))
            cost_savings = battery_reduction * request.battery_cost_per_kwh
            extension = float(max(0.0, request.mission_duration_days * 0.05 if surplus > 0 else 0.0))
            mission_impacts = MissionImpacts(
                power_generation_kwh_per_day=generation,
                power_surplus_kwh_per_day=surplus,
                mission_duration_extension_days=extension,
                cost_savings_usd=cost_savings,
                battery_reduction_kwh=battery_reduction,
            )

        # Convert numpy arrays to lists for JSON serialization
        potential_list = solar_result.solar_potential_map.tolist()
        irradiance_list = solar_result.irradiance_map.tolist()

        # Replace NaN with 0 for JSON serialization
        def clean_nan(arr):
            if isinstance(arr, list):
                return [clean_nan(item) if isinstance(item, (list, np.ndarray)) else (0.0 if (isinstance(item, float) and np.isnan(item)) else item) for item in arr]
            elif isinstance(arr, np.ndarray):
                arr = np.nan_to_num(arr, nan=0.0)
                return arr.tolist()
            return arr

        potential_list = clean_nan(potential_list)
        irradiance_list = clean_nan(irradiance_list)

        return SolarAnalysisResponse(
            solar_potential_map=potential_list,
            irradiance_map=irradiance_list,
            statistics=SolarStatistics(**solar_result.statistics),
            mission_impacts=MissionImpactsResponse(
                power_generation_kwh_per_day=mission_impacts.power_generation_kwh_per_day,
                power_surplus_kwh_per_day=mission_impacts.power_surplus_kwh_per_day,
                mission_duration_extension_days=mission_impacts.mission_duration_extension_days,
                cost_savings_usd=mission_impacts.cost_savings_usd,
                battery_reduction_kwh=mission_impacts.battery_reduction_kwh
            ),
            shape={"rows": solar_result.solar_potential_map.shape[0], "cols": solar_result.solar_potential_map.shape[1]}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Solar analysis failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail={"error":"solar_failed","detail":str(e), "hint":"Reduce ROI size and ensure DEM cached or enable demo mode."})

