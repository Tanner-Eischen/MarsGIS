"""Coordinate transformations for Mars reference frames."""

from typing import Tuple

import numpy as np
try:
    import spiceypy as spice
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False
    import warnings
    warnings.warn("SpiceyPy not available, using simplified transforms")

from marshab.exceptions import CoordinateError
from marshab.types import SiteOrigin
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class CoordinateTransformer:
    """Transforms between Mars coordinate reference frames."""
    
    def __init__(
        self,
        equatorial_radius: float = 3396190.0,
        polar_radius: float = 3376200.0,
        use_spice: bool = True
    ):
        """Initialize coordinate transformer.
        
        Args:
            equatorial_radius: Mars equatorial radius (meters)
            polar_radius: Mars polar radius (meters)
            use_spice: Whether to use SPICE toolkit if available
        """
        self.eq_radius = equatorial_radius
        self.pol_radius = polar_radius
        self.use_spice = use_spice and SPICE_AVAILABLE
        
        if not self.use_spice:
            logger.warning("Using simplified coordinate transforms (SPICE not available)")
    
    def planetocentric_to_cartesian(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float
    ) -> Tuple[float, float, float]:
        """Convert planetocentric lat/lon to Cartesian coordinates.
        
        Args:
            lat_deg: Planetocentric latitude (degrees)
            lon_deg: East positive longitude (degrees)
            elevation_m: Elevation above datum (meters)
        
        Returns:
            (x, y, z) in Mars body-fixed frame (meters)
        """
        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        
        # Calculate radius at this latitude (ellipsoid)
        # r = sqrt((a^2 * cos(lat))^2 + (b^2 * sin(lat))^2) / sqrt((a*cos(lat))^2 + (b*sin(lat))^2)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        
        numerator = np.sqrt(
            (self.eq_radius**2 * cos_lat)**2 + (self.pol_radius**2 * sin_lat)**2
        )
        denominator = np.sqrt(
            (self.eq_radius * cos_lat)**2 + (self.pol_radius * sin_lat)**2
        )
        
        radius = numerator / denominator + elevation_m
        
        # Convert to Cartesian
        x = radius * cos_lat * np.cos(lon_rad)
        y = radius * cos_lat * np.sin(lon_rad)
        z = radius * sin_lat
        
        return x, y, z
    
    def iau_mars_to_site_frame(
        self,
        lat_deg: float,
        lon_deg: float,
        elevation_m: float,
        site_origin: SiteOrigin
    ) -> Tuple[float, float, float]:
        """Transform IAU_MARS coordinates to rover SITE frame.
        
        SITE frame definition:
        - Origin: site_origin location on Mars surface
        - +X axis: North
        - +Y axis: East
        - +Z axis: Down (nadir)
        
        Args:
            lat_deg: Target latitude (degrees, planetocentric)
            lon_deg: Target longitude (degrees, east positive)
            elevation_m: Target elevation (meters above datum)
            site_origin: SITE frame origin definition
        
        Returns:
            (x, y, z) in SITE frame (meters)
        
        Raises:
            CoordinateError: If transformation fails
        """
        try:
            # Convert both points to Cartesian
            target_xyz = self.planetocentric_to_cartesian(lat_deg, lon_deg, elevation_m)
            origin_xyz = self.planetocentric_to_cartesian(
                site_origin.lat,
                site_origin.lon,
                site_origin.elevation_m
            )
            
            # Calculate offset vector
            dx = target_xyz[0] - origin_xyz[0]
            dy = target_xyz[1] - origin_xyz[1]
            dz = target_xyz[2] - origin_xyz[2]
            
            # Build rotation matrix from Mars-fixed to local NED frame
            # (Simplified approach - for production use proper DCM)
            lat_rad = np.radians(site_origin.lat)
            lon_rad = np.radians(site_origin.lon)
            
            # Rotation matrix: Mars-fixed (XYZ) -> Local NED (North-East-Down)
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)
            
            # DCM from ECEF to NED
            R = np.array([
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],   # North
                [-sin_lon, cos_lon, 0],                               # East
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]   # Down
            ])
            
            # Apply rotation
            offset_ned = R @ np.array([dx, dy, dz])
            
            x_site = float(offset_ned[0])  # Northing
            y_site = float(offset_ned[1])  # Easting
            z_site = float(offset_ned[2])  # Down
            
            logger.debug(
                "Transformed to SITE frame",
                target_lat=lat_deg,
                target_lon=lon_deg,
                site_x=x_site,
                site_y=y_site
            )
            
            return x_site, y_site, z_site
            
        except Exception as e:
            raise CoordinateError(
                "Failed to transform coordinates to SITE frame",
                details={
                    "lat": lat_deg,
                    "lon": lon_deg,
                    "elevation_m": elevation_m,
                    "site_origin_lat": site_origin.lat,
                    "site_origin_lon": site_origin.lon,
                    "error": str(e)
                }
            )
    
    def site_frame_to_iau_mars(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        site_origin: SiteOrigin
    ) -> Tuple[float, float, float]:
        """Transform SITE frame coordinates back to IAU_MARS.
        
        Args:
            x_m: X coordinate in SITE frame (North, meters)
            y_m: Y coordinate in SITE frame (East, meters)
            z_m: Z coordinate in SITE frame (Down, meters)
            site_origin: SITE frame origin
        
        Returns:
            (lat, lon, elevation) in IAU_MARS (degrees, degrees, meters)
        """
        # This is the inverse of iau_mars_to_site_frame
        # Implementation left as exercise - requires inverse rotation
        # and Cartesian to geodetic conversion
        
        raise NotImplementedError("Inverse transform not yet implemented")

