"""Unit tests for type definitions and models."""

import pytest
from pydantic import ValidationError

from marshab.types import (
    BoundingBox,
    SiteOrigin,
    Waypoint,
    CriteriaWeights,
    AnalysisConfig,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_valid_bbox(self):
        """Test creating valid bounding box."""
        bbox = BoundingBox(
            lat_min=-45,
            lat_max=45,
            lon_min=0,
            lon_max=360,
        )
        assert bbox.lat_min == -45
        assert bbox.lat_max == 45

    def test_bbox_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BoundingBox(lat_min=10, lat_max=20, lon_min=30, lon_max=40)
        assert bbox.to_tuple() == (10, 20, 30, 40)

    def test_invalid_lat_range(self):
        """Test validation of latitude range."""
        with pytest.raises(ValidationError):
            BoundingBox(lat_min=45, lat_max=45, lon_min=0, lon_max=10)

    def test_lat_bounds(self):
        """Test latitude bounds validation."""
        with pytest.raises(ValidationError):
            BoundingBox(lat_min=-91, lat_max=0, lon_min=0, lon_max=10)


class TestSiteOrigin:
    """Tests for SiteOrigin model."""

    def test_valid_site_origin(self):
        """Test creating valid site origin."""
        origin = SiteOrigin(lat=40.5, lon=180.5, elevation_m=-2500)
        assert origin.lat == 40.5
        assert origin.elevation_m == -2500

    def test_lon_bounds(self):
        """Test longitude bounds."""
        with pytest.raises(ValidationError):
            SiteOrigin(lat=0, lon=361, elevation_m=0)


class TestWaypoint:
    """Tests for Waypoint model."""

    def test_valid_waypoint(self):
        """Test creating valid waypoint."""
        wp = Waypoint(
            waypoint_id=0,
            x_meters=100.0,
            y_meters=50.0,
            tolerance_meters=5.0,
        )
        assert wp.waypoint_id == 0
        assert wp.distance_from_origin() == pytest.approx(111.80, rel=0.01)

    def test_waypoint_at_origin(self):
        """Test waypoint at origin."""
        wp = Waypoint(waypoint_id=1, x_meters=0, y_meters=0)
        assert wp.distance_from_origin() == 0

    def test_waypoint_with_heading(self):
        """Test waypoint with heading."""
        wp = Waypoint(
            waypoint_id=2,
            x_meters=100.0,
            y_meters=50.0,
            heading_deg=45.0,
        )
        assert wp.heading_deg == 45.0


class TestCriteriaWeights:
    """Tests for CriteriaWeights model."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = CriteriaWeights()
        assert weights.slope == 0.30
        assert weights.roughness == 0.25
        assert weights.elevation == 0.20
        assert weights.total() == 1.0

    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = CriteriaWeights(
            slope=3, roughness=2.5, elevation=2, solar_exposure=1.5, resources=1
        )
        normalized = weights.normalize()
        assert normalized.total() == pytest.approx(1.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = CriteriaWeights()
        d = weights.to_dict()
        assert isinstance(d, dict)
        assert "slope" in d

    def test_normalize_zero_total(self):
        """Test normalization with zero total raises error."""
        weights = CriteriaWeights(
            slope=0, roughness=0, elevation=0, solar_exposure=0, resources=0
        )
        with pytest.raises(ValueError, match="Total weight cannot be zero"):
            weights.normalize()


class TestAnalysisConfig:
    """Tests for AnalysisConfig model."""

    def test_valid_config(self, test_roi):
        """Test creating valid analysis config."""
        config = AnalysisConfig(
            roi=test_roi,
            max_slope_deg=10,
            suitability_threshold=0.75,
        )
        assert config.roi == test_roi
        assert config.max_slope_deg == 10
        assert config.suitability_threshold == 0.75
        assert isinstance(config.criteria_weights, CriteriaWeights)

    def test_default_criteria_weights(self, test_roi):
        """Test default criteria weights are used."""
        config = AnalysisConfig(roi=test_roi)
        assert config.criteria_weights.total() == 1.0

