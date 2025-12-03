"""Solar potential analysis for Mars terrain."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from marshab.processing.criteria import CriteriaExtractor
from marshab.processing.terrain import TerrainAnalyzer
from marshab.models import TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

# Mars solar constant (W/m²) - approximately 590 W/m² at Mars distance
MARS_SOLAR_CONSTANT = 590.0

# Typical Mars solar panel efficiency
DEFAULT_PANEL_EFFICIENCY = 0.25

# Dust accumulation factor (reduces efficiency over time)
DUST_DEGRADATION_FACTOR = 0.85


@dataclass
class SolarPotentialResult:
    """Results from solar potential analysis."""
    solar_potential_map: np.ndarray  # Normalized 0-1
    irradiance_map: np.ndarray  # W/m²
    statistics: Dict[str, float]
    mission_impacts: Dict[str, float]


@dataclass
class MissionImpacts:
    """Mission impact calculations."""
    power_generation_kwh_per_day: float
    power_surplus_kwh_per_day: float
    mission_duration_extension_days: float
    cost_savings_usd: float
    battery_reduction_kwh: float


class SolarPotentialAnalyzer:
    """Analyzes solar potential from terrain data."""
    
    def __init__(self, cell_size_m: float = 200.0):
        """Initialize solar potential analyzer.
        
        Args:
            cell_size_m: DEM cell size in meters
        """
        self.cell_size_m = cell_size_m
        logger.info("Initialized SolarPotentialAnalyzer", cell_size_m=cell_size_m)
    
    def calculate_solar_potential(
        self,
        dem: xr.DataArray,
        metrics: TerrainMetrics,
        sun_azimuth: float = 0.0,
        sun_altitude: float = 45.0,
        dust_cover_index: Optional[xr.DataArray] = None
    ) -> SolarPotentialResult:
        """Calculate solar potential from DEM and terrain metrics.
        
        Args:
            dem: DEM DataArray
            metrics: TerrainMetrics from terrain analysis
            sun_azimuth: Sun azimuth angle in degrees (0-360, 0=North)
            sun_altitude: Sun altitude angle in degrees (0-90, 0=horizon)
            dust_cover_index: Optional TES Dust Cover Index DataArray for spatially variable dust degradation
            
        Returns:
            SolarPotentialResult with potential map and statistics
        """
        logger.info(
            "Calculating solar potential",
            sun_azimuth=sun_azimuth,
            sun_altitude=sun_altitude
        )
        
        elevation = dem.values.astype(np.float32)
        
        # Get solar exposure from criteria extractor
        extractor = CriteriaExtractor(dem, metrics)
        solar_exposure = extractor.calculate_solar_exposure()
        
        # Calculate shadow penalty (inverse of illumination)
        shadow_penalty = self._calculate_shadow_penalty(
            elevation,
            sun_azimuth,
            sun_altitude
        )
        
        # Combine solar exposure with shadow information
        # Shadow reduces potential: potential = exposure * (1 - shadow_penalty)
        solar_potential = solar_exposure * (1.0 - shadow_penalty * 0.5)  # Shadow reduces by up to 50%
        solar_potential = np.clip(solar_potential, 0.0, 1.0)
        
        # Calculate actual irradiance (W/m²)
        # Account for sun altitude: lower altitude = less irradiance
        altitude_factor = np.sin(np.radians(sun_altitude))
        altitude_factor = np.clip(altitude_factor, 0.0, 1.0)
        
        base_irradiance = MARS_SOLAR_CONSTANT * altitude_factor
        irradiance_map = base_irradiance * solar_potential
        
        # Apply spatially variable dust degradation if DCI data is available
        if dust_cover_index is not None:
            try:
                # Resample DCI to match DEM resolution if needed
                if dust_cover_index.shape != dem.shape:
                    from scipy import ndimage
                    scale_factor = dem.shape[0] / dust_cover_index.shape[0]
                    dci_resampled = ndimage.zoom(
                        dust_cover_index.values,
                        (scale_factor, scale_factor),
                        order=1
                    )
                    # Crop/pad to match exact shape
                    if dci_resampled.shape != dem.shape:
                        dci_resampled = dci_resampled[:dem.shape[0], :dem.shape[1]]
                else:
                    dci_resampled = dust_cover_index.values
                
                # Convert DCI to dust degradation factor
                # DCI: lower emissivity (1350-1400 cm⁻¹) = higher dust = lower efficiency
                # Normalize DCI to 0-1 range (assuming typical range)
                dci_normalized = np.clip((dci_resampled - 0.85) / 0.15, 0.0, 1.0)  # Typical range: 0.85-1.0
                
                # Convert to degradation factor: high dust (low DCI) = low efficiency
                # Range: 0.70 (high dust) to 0.95 (low dust)
                dust_degradation_map = 0.70 + (1.0 - dci_normalized) * 0.25
                
                # Apply dust degradation to irradiance
                irradiance_map = irradiance_map * dust_degradation_map
                
                logger.info(
                    "Applied spatially variable dust degradation",
                    mean_degradation=float(np.nanmean(dust_degradation_map)),
                    min_degradation=float(np.nanmin(dust_degradation_map)),
                    max_degradation=float(np.nanmax(dust_degradation_map))
                )
            except Exception as e:
                logger.warning(
                    "Failed to apply DCI dust degradation, using global factor",
                    error=str(e)
                )
                # Fall back to global degradation factor
                irradiance_map = irradiance_map * DUST_DEGRADATION_FACTOR
        else:
            # Use global dust degradation factor
            irradiance_map = irradiance_map * DUST_DEGRADATION_FACTOR
        
        # Calculate statistics
        valid_mask = ~np.isnan(solar_potential)
        if np.any(valid_mask):
            stats = {
                "min": float(np.nanmin(solar_potential)),
                "max": float(np.nanmax(solar_potential)),
                "mean": float(np.nanmean(solar_potential)),
                "std": float(np.nanstd(solar_potential)),
                "min_irradiance_w_per_m2": float(np.nanmin(irradiance_map)),
                "max_irradiance_w_per_m2": float(np.nanmax(irradiance_map)),
                "mean_irradiance_w_per_m2": float(np.nanmean(irradiance_map)),
            }
        else:
            stats = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min_irradiance_w_per_m2": 0.0,
                "max_irradiance_w_per_m2": 0.0,
                "mean_irradiance_w_per_m2": 0.0,
            }
        
        # Placeholder mission impacts (will be calculated with panel config)
        mission_impacts = {
            "power_generation_kwh_per_day": 0.0,
            "power_surplus_kwh_per_day": 0.0,
            "mission_duration_extension_days": 0.0,
            "cost_savings_usd": 0.0,
            "battery_reduction_kwh": 0.0,
        }
        
        return SolarPotentialResult(
            solar_potential_map=solar_potential,
            irradiance_map=irradiance_map,
            statistics=stats,
            mission_impacts=mission_impacts
        )
    
    def calculate_mission_impacts(
        self,
        solar_result: SolarPotentialResult,
        panel_efficiency: float = DEFAULT_PANEL_EFFICIENCY,
        panel_area_m2: float = 100.0,
        battery_capacity_kwh: float = 50.0,
        daily_power_needs_kwh: float = 20.0,
        battery_cost_per_kwh: float = 1000.0,
        mission_duration_days: float = 500.0
    ) -> MissionImpacts:
        """Calculate mission impact from solar potential.
        
        Args:
            solar_result: SolarPotentialResult from calculate_solar_potential
            panel_efficiency: Solar panel efficiency (0-1)
            panel_area_m2: Total panel area in square meters
            battery_capacity_kwh: Baseline battery capacity
            daily_power_needs_kwh: Daily power requirements
            battery_cost_per_kwh: Cost per kWh of battery storage
            mission_duration_days: Baseline mission duration
            
        Returns:
            MissionImpacts with power, duration, and cost calculations
        """
        logger.info(
            "Calculating mission impacts",
            panel_efficiency=panel_efficiency,
            panel_area_m2=panel_area_m2
        )
        
        # Use mean irradiance for calculations
        mean_irradiance = solar_result.statistics["mean_irradiance_w_per_m2"]
        
        # Account for dust degradation
        effective_irradiance = mean_irradiance * DUST_DEGRADATION_FACTOR
        
        # Calculate power generation (kWh/day)
        # Assume 6 hours of effective sunlight per day (Mars day is ~24.6 hours)
        effective_sunlight_hours = 6.0
        power_generation_kw = (effective_irradiance * panel_area_m2 * panel_efficiency) / 1000.0
        power_generation_kwh_per_day = power_generation_kw * effective_sunlight_hours
        
        # Calculate power surplus
        power_surplus_kwh_per_day = power_generation_kwh_per_day - daily_power_needs_kwh
        
        # Calculate mission duration extension
        # If we have surplus, we can extend mission by reducing battery dependency
        if power_surplus_kwh_per_day > 0:
            # Additional days from surplus power
            # Assume surplus can extend mission by reducing battery drain
            surplus_per_mission = power_surplus_kwh_per_day * mission_duration_days
            extension_days = surplus_per_mission / daily_power_needs_kwh
        else:
            extension_days = 0.0
        
        # Calculate battery reduction
        # If we generate more than needed, we can reduce battery capacity
        if power_surplus_kwh_per_day > 0:
            # Reduce battery by amount we can generate in reserve
            battery_reduction_kwh = min(
                power_surplus_kwh_per_day * 2.0,  # 2 days of surplus as buffer
                battery_capacity_kwh * 0.3  # Max 30% reduction
            )
        else:
            battery_reduction_kwh = 0.0
        
        # Calculate cost savings
        # Battery cost savings + extended mission value
        battery_cost_savings = battery_reduction_kwh * battery_cost_per_kwh
        
        # Extended mission value (simplified: $10k per day)
        mission_value_per_day = 10000.0
        mission_value = extension_days * mission_value_per_day
        
        cost_savings_usd = battery_cost_savings + mission_value
        
        return MissionImpacts(
            power_generation_kwh_per_day=power_generation_kwh_per_day,
            power_surplus_kwh_per_day=power_surplus_kwh_per_day,
            mission_duration_extension_days=extension_days,
            cost_savings_usd=cost_savings_usd,
            battery_reduction_kwh=battery_reduction_kwh
        )
    
    def _calculate_shadow_penalty(
        self,
        elevation: np.ndarray,
        sun_azimuth: float,
        sun_altitude: float
    ) -> np.ndarray:
        """Calculate shadow penalty based on terrain and sun position.
        
        Uses simplified approach from routing.py shadow calculation.
        
        Args:
            elevation: Elevation array (meters)
            sun_azimuth: Sun azimuth angle in degrees (0-360, 0=North)
            sun_altitude: Sun altitude angle in degrees (0-90, 0=horizon)
            
        Returns:
            Shadow penalty array (0 = no shadow, 1 = full shadow)
        """
        if elevation.size == 0:
            return np.zeros_like(elevation)
        
        # Convert to radians
        azimuth_rad = np.deg2rad(sun_azimuth)
        altitude_rad = np.deg2rad(sun_altitude)
        
        # Calculate gradients (slope and aspect)
        dy, dx = np.gradient(elevation, self.cell_size_m)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dx, dy)  # Aspect: 0=North, positive=east
        
        # Calculate sun direction vector
        sun_x = np.sin(azimuth_rad) * np.cos(altitude_rad)
        sun_y = np.cos(azimuth_rad) * np.cos(altitude_rad)
        sun_z = np.sin(altitude_rad)
        
        # Calculate surface normal
        normal_x = -np.sin(aspect) * np.sin(slope)
        normal_y = -np.cos(aspect) * np.sin(slope)
        normal_z = np.cos(slope)
        
        # Dot product: negative means surface faces away from sun (in shadow)
        dot_product = normal_x * sun_x + normal_y * sun_y + normal_z * sun_z
        
        # Convert to shadow penalty (0-1)
        # Negative dot product = shadow, positive = illuminated
        shadow_penalty = np.clip(-dot_product, 0, 1)
        
        return shadow_penalty.astype(np.float32)

