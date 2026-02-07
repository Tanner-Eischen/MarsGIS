"""Load and manage criteria presets from configuration."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, field_validator

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class PresetWeights(BaseModel):
    """Weights for a preset."""
    slope: Optional[float] = None
    roughness: Optional[float] = None
    elevation: Optional[float] = None
    solar_exposure: Optional[float] = None
    science_value: Optional[float] = None
    distance: Optional[float] = None
    slope_penalty: Optional[float] = None
    roughness_penalty: Optional[float] = None
    elevation_penalty: Optional[float] = None

    @field_validator('*')
    @classmethod
    def validate_weight_range(cls, v):
        """Ensure weights are in valid range."""
        if v is not None and not 0 <= v <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {v}")
        return v


class PresetThresholds(BaseModel):
    """Thresholds for a preset."""
    max_slope_deg: Optional[float] = None
    max_roughness: Optional[float] = None
    min_site_area_km2: Optional[float] = 0.5


class Preset(BaseModel):
    """Single preset configuration."""
    id: str
    name: str
    description: str
    scope: Literal["site", "route"]
    weights: PresetWeights
    thresholds: PresetThresholds

    def get_weights_dict(self) -> dict[str, float]:
        """Get non-None weights as dictionary."""
        return {
            k: v for k, v in self.weights.model_dump().items()
            if v is not None
        }

    def validate_weights_sum(self) -> bool:
        """Check if weights sum to approximately 1.0."""
        weights = self.get_weights_dict()
        total = sum(weights.values())
        return 0.99 <= total <= 1.01


class CriterionDefinition(BaseModel):
    """Criterion metadata."""
    display_name: str
    description: str
    unit: str
    beneficial: bool
    min_value: Optional[float]
    max_value: Optional[float]


class PresetsConfig(BaseModel):
    """Complete presets configuration."""
    site_presets: dict[str, Preset]
    route_presets: dict[str, Preset]
    criteria: dict[str, CriterionDefinition]


class PresetLoader:
    """Loads and manages preset configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize preset loader.

        Args:
            config_path: Optional path to presets YAML file
        """
        if config_path is None:
            # Default location
            config_path = Path(__file__).parent / "criteria_presets.yaml"

        self.config_path = config_path
        self.config: Optional[PresetsConfig] = None

        if self.config_path.exists():
            self.load()
        else:
            logger.warning(
                f"Presets config not found: {config_path}",
                "Using minimal defaults"
            )

    def load(self) -> PresetsConfig:
        """Load presets from YAML file.

        Returns:
            Loaded PresetsConfig
        """
        logger.info(f"Loading presets from {self.config_path}")

        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        # Parse into Pydantic models
        self.config = PresetsConfig(
            site_presets={
                k: Preset(**{**v, "scope": "site"})
                for k, v in data.get('site_presets', {}).items()
            },
            route_presets={
                k: Preset(**{**v, "scope": "route"})
                for k, v in data.get('route_presets', {}).items()
            },
            criteria={
                k: CriterionDefinition(**v)
                for k, v in data.get('criteria', {}).items()
            }
        )

        # Validate presets
        for preset_id, preset in self.config.site_presets.items():
            if not preset.validate_weights_sum():
                logger.warning(
                    f"Site preset weights don't sum to 1.0: {preset_id}",
                    weights=preset.get_weights_dict()
                )

        for preset_id, preset in self.config.route_presets.items():
            if not preset.validate_weights_sum():
                logger.warning(
                    f"Route preset weights don't sum to 1.0: {preset_id}",
                    weights=preset.get_weights_dict()
                )

        logger.info(
            "Presets loaded",
            site_presets=len(self.config.site_presets),
            route_presets=len(self.config.route_presets),
            criteria=len(self.config.criteria)
        )

        return self.config

    def get_preset(
        self,
        preset_id: str,
        scope: Literal["site", "route"]
    ) -> Optional[Preset]:
        """Get preset by ID and scope.

        Args:
            preset_id: Preset identifier
            scope: "site" or "route"

        Returns:
            Preset object or None if not found
        """
        if self.config is None:
            return None

        presets = (
            self.config.site_presets if scope == "site"
            else self.config.route_presets
        )

        return presets.get(preset_id)

    def list_presets(
        self,
        scope: Optional[Literal["site", "route"]] = None
    ) -> list[Preset]:
        """List available presets.

        Args:
            scope: Optional filter by scope

        Returns:
            List of Preset objects
        """
        if self.config is None:
            return []

        presets = []

        if scope is None or scope == "site":
            presets.extend(self.config.site_presets.values())

        if scope is None or scope == "route":
            presets.extend(self.config.route_presets.values())

        return presets

    def get_criterion(self, criterion_name: str) -> Optional[CriterionDefinition]:
        """Get criterion definition.

        Args:
            criterion_name: Criterion identifier

        Returns:
            CriterionDefinition or None
        """
        if self.config is None:
            return None

        return self.config.criteria.get(criterion_name)

