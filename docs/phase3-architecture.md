# Phase 3: Decision Lab Interface - Architecture & Implementation Plan

**Duration:** 3-4 weeks  
**Priority:** MEDIUM  
**Goal:** Implement preset-based criteria exploration UI with progressive disclosure

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                       Decision Lab                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Frontend (React)              Backend (FastAPI)                 │
│  ┌──────────────┐             ┌──────────────┐                  │
│  │   Decision   │────API─────▶│   Presets    │                  │
│  │   Lab Page   │             │   Loader     │                  │
│  └──────────────┘             └──────────────┘                  │
│         │                             │                          │
│         │                             ▼                          │
│         │                      ┌──────────────┐                  │
│         │                      │    Site      │                  │
│         ├────────────API──────▶│   Scoring    │                  │
│         │                      │   Engine     │                  │
│         │                      └──────────────┘                  │
│         │                             │                          │
│         │                             ▼                          │
│         │                      ┌──────────────┐                  │
│         └────────────API──────▶│    Route     │                  │
│                                │ Cost Engine  │                  │
│                                └──────────────┘                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### User Experience Flow

1. **Preset Selection** → User picks site/route preset (Safe, Balanced, Science-focused)
2. **Auto-Analysis** → System computes scores with preset weights
3. **Results Display** → Ranked sites/routes with visual explanations
4. **Progressive Disclosure** → Optional: Open advanced panel, adjust weights
5. **Re-computation** → System recalculates with custom weights
6. **Visualization** → 3D terrain with overlays, sun lighting controls

---

## Task 1: Preset Configuration System

### Objective
Create YAML-based preset definitions and loader system.

### Implementation

#### 1.1 Preset Configuration File

**File:** `marshab/config/criteria_presets.yaml` (new)

```yaml
# Mars Landing Site & Route Decision Lab - Preset Configurations

site_presets:
  safe_landing:
    id: safe_landing
    name: "Safe Landing"
    description: "Prioritizes landing safety with gentle slopes and smooth terrain. Ideal for risk-averse missions."
    scope: site
    weights:
      slope: 0.40          # High emphasis on gentle slopes
      roughness: 0.30      # Smooth surface critical
      elevation: 0.15      # Lower elevations preferred (atmosphere)
      solar_exposure: 0.10 # Some solar consideration
      science_value: 0.05  # Lower priority
    thresholds:
      max_slope_deg: 5.0
      max_roughness: 0.3
  
  balanced:
    id: balanced
    name: "Balanced Mission"
    description: "Balanced approach considering safety, energy, and science objectives equally."
    scope: site
    weights:
      slope: 0.25
      roughness: 0.20
      elevation: 0.20
      solar_exposure: 0.20
      science_value: 0.15
    thresholds:
      max_slope_deg: 8.0
      max_roughness: 0.5
  
  science_focused:
    id: science_focused
    name: "Science-Focused"
    description: "Maximizes scientific value with proximity to features of interest. Accepts higher terrain complexity."
    scope: site
    weights:
      slope: 0.15
      roughness: 0.15
      elevation: 0.10
      solar_exposure: 0.15
      science_value: 0.45  # Heavy emphasis on science
    thresholds:
      max_slope_deg: 12.0
      max_roughness: 0.7

route_presets:
  shortest_path:
    id: shortest_path
    name: "Shortest Path"
    description: "Minimizes travel distance. Safety constraints enforced but distance prioritized."
    scope: route
    weights:
      distance: 0.50        # Primary objective
      slope_penalty: 0.20   # Basic safety
      roughness_penalty: 0.15
      elevation_penalty: 0.15
    thresholds:
      max_slope_deg: 25.0
      max_roughness: 1.0
  
  safest_path:
    id: safest_path
    name: "Safest Path"
    description: "Minimizes terrain hazards even if path is longer. Ideal for valuable rovers."
    scope: route
    weights:
      distance: 0.15
      slope_penalty: 0.40   # High safety emphasis
      roughness_penalty: 0.30
      elevation_penalty: 0.15
    thresholds:
      max_slope_deg: 15.0
      max_roughness: 0.5
  
  energy_optimal:
    id: energy_optimal
    name: "Energy-Optimal"
    description: "Balances distance and terrain to minimize energy consumption for solar-powered rovers."
    scope: route
    weights:
      distance: 0.30
      slope_penalty: 0.25
      roughness_penalty: 0.20
      elevation_penalty: 0.25  # Avoid climbing
    thresholds:
      max_slope_deg: 20.0
      max_roughness: 0.6

# Criterion definitions
criteria:
  slope:
    display_name: "Slope Safety"
    description: "Terrain slope angle. Lower slopes are safer for landing and traversing."
    unit: "degrees"
    beneficial: false
    min_value: 0
    max_value: 90
    
  roughness:
    display_name: "Surface Roughness"
    description: "Local terrain variation. Smoother surfaces reduce landing risk and improve traversability."
    unit: "meters"
    beneficial: false
    min_value: 0
    max_value: null
    
  elevation:
    display_name: "Elevation"
    description: "Height above Mars datum. Lower elevations have better atmospheric density for landing."
    unit: "meters"
    beneficial: false
    min_value: null
    max_value: null
    
  solar_exposure:
    display_name: "Solar Exposure"
    description: "Estimated solar energy availability based on slope and aspect. Higher is better for power generation."
    unit: "normalized"
    beneficial: true
    min_value: 0
    max_value: 1
    
  science_value:
    display_name: "Science Value"
    description: "Proximity to features of scientific interest (craters, minerals, ice deposits)."
    unit: "normalized"
    beneficial: true
    min_value: 0
    max_value: 1
```

#### 1.2 Preset Loader

**File:** `marshab/config/preset_loader.py` (new)

```python
"""Load and manage criteria presets from configuration."""

from pathlib import Path
from typing import Dict, List, Optional, Literal
import yaml
from pydantic import BaseModel, Field, validator

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
    
    @validator('*')
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
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Get non-None weights as dictionary."""
        return {
            k: v for k, v in self.weights.dict().items() 
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
    site_presets: Dict[str, Preset]
    route_presets: Dict[str, Preset]
    criteria: Dict[str, CriterionDefinition]


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
        
        with open(self.config_path, 'r') as f:
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
    ) -> List[Preset]:
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
```

### Testing

**File:** `tests/unit/test_preset_loader.py`

```python
"""Tests for preset loader."""

import pytest
from pathlib import Path
from marshab.config.preset_loader import PresetLoader, Preset

class TestPresetLoader:
    """Tests for PresetLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Provide PresetLoader with test config."""
        return PresetLoader()
    
    def test_load_presets(self, loader):
        """Test loading presets from YAML."""
        config = loader.load()
        
        assert config is not None
        assert len(config.site_presets) > 0
        assert len(config.route_presets) > 0
        assert len(config.criteria) > 0
    
    def test_get_site_preset(self, loader):
        """Test retrieving site preset."""
        loader.load()
        preset = loader.get_preset("safe_landing", "site")
        
        assert preset is not None
        assert preset.name == "Safe Landing"
        assert preset.scope == "site"
        assert "slope" in preset.get_weights_dict()
    
    def test_weights_validation(self, loader):
        """Test preset weights sum validation."""
        loader.load()
        preset = loader.get_preset("balanced", "site")
        
        assert preset.validate_weights_sum()
    
    def test_list_presets(self, loader):
        """Test listing all presets."""
        loader.load()
        
        all_presets = loader.list_presets()
        assert len(all_presets) > 0
        
        site_presets = loader.list_presets(scope="site")
        route_presets = loader.list_presets(scope="route")
        
        assert len(site_presets) > 0
        assert len(route_presets) > 0
```

---

## Task 2: Site Analysis API

### Objective
Implement API endpoints for site scoring with presets and custom weights.

### Implementation

**File:** `marshab/analysis/site_scoring.py` (new)

```python
"""Site scoring with MCDM and explainability."""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from marshab.processing.mcdm import MCDMEvaluator
from marshab.processing.criteria import CriteriaExtractor
from marshab.types import Site, TerrainMetrics
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SiteScore:
    """Site score with component breakdown."""
    site: Site
    total_score: float
    components: Dict[str, float]  # Criterion name -> contribution
    explanation: str


class SiteScoringEngine:
    """Scores sites using MCDM with explainability."""
    
    def score_sites(
        self,
        sites: List[Site],
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> List[SiteScore]:
        """Score sites and provide component breakdowns.
        
        Args:
            sites: List of candidate sites
            criteria: Criterion arrays
            weights: Criterion weights
            beneficial: Benefit directions
            
        Returns:
            List of SiteScore objects with explanations
        """
        logger.info(f"Scoring {len(sites)} sites with MCDM")
        
        # Compute suitability surface
        suitability = MCDMEvaluator.evaluate(
            criteria, weights, beneficial, method="weighted_sum"
        )
        
        scored_sites = []
        
        for site in sites:
            # Extract site region from criteria
            # (Simplified - assumes site mask available)
            site_components = {}
            
            for criterion_name, values in criteria.items():
                # Get mean value over site region
                # In real implementation, use site.geometry to mask
                site_components[criterion_name] = float(np.mean(values))
            
            # Generate explanation
            explanation = self._generate_explanation(
                site_components, weights, beneficial
            )
            
            scored_sites.append(SiteScore(
                site=site,
                total_score=site.suitability_score,
                components=site_components,
                explanation=explanation
            ))
        
        # Sort by total score
        scored_sites.sort(key=lambda x: x.total_score, reverse=True)
        
        return scored_sites
    
    def _generate_explanation(
        self,
        components: Dict[str, float],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> str:
        """Generate plain-language explanation.
        
        Args:
            components: Criterion values
            weights: Criterion weights
            beneficial: Benefit directions
            
        Returns:
            Human-readable explanation string
        """
        explanations = []
        
        # Categorize criteria
        for criterion, value in components.items():
            weight = weights.get(criterion, 0)
            is_beneficial = beneficial.get(criterion, True)
            
            if weight < 0.05:
                continue  # Skip low-weight criteria
            
            # Qualitative assessment
            if criterion == "slope":
                if value < 3:
                    explanations.append("very gentle slopes (excellent)")
                elif value < 7:
                    explanations.append("moderate slopes (good)")
                else:
                    explanations.append("steeper slopes (acceptable)")
            
            elif criterion == "roughness":
                if value < 0.3:
                    explanations.append("smooth terrain (excellent)")
                elif value < 0.6:
                    explanations.append("moderate roughness (good)")
                else:
                    explanations.append("rough terrain (acceptable)")
            
            elif criterion == "solar_exposure":
                if value > 0.7:
                    explanations.append("excellent solar exposure")
                elif value > 0.4:
                    explanations.append("adequate solar exposure")
                else:
                    explanations.append("limited solar exposure")
            
            elif criterion == "science_value":
                if value > 0.7:
                    explanations.append("high scientific interest")
                elif value > 0.4:
                    explanations.append("moderate scientific value")
        
        if not explanations:
            return "This site meets basic criteria."
        
        return "This site has " + ", ".join(explanations) + "."
```

**File:** `marshab/web/routes/site_analysis.py` (new)

```python
"""Site analysis API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List

from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.analysis.site_scoring import SiteScoringEngine
from marshab.config.preset_loader import PresetLoader
from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class SiteAnalysisRequest(BaseModel):
    """Request for site analysis."""
    roi: Dict[str, float] = Field(
        ..., 
        description="Bounding box: lat_min, lat_max, lon_min, lon_max"
    )
    dataset: str = Field("mola", description="DEM dataset")
    preset_id: Optional[str] = Field(None, description="Preset ID to use")
    custom_weights: Optional[Dict[str, float]] = Field(
        None, 
        description="Custom criterion weights (overrides preset)"
    )
    threshold: float = Field(0.7, ge=0, le=1)


class SiteScoreResponse(BaseModel):
    """Single site score response."""
    site_id: int
    rank: int
    total_score: float
    components: Dict[str, float]
    explanation: str
    geometry: dict
    centroid_lat: float
    centroid_lon: float
    area_km2: float


@router.post("/site-scores", response_model=List[SiteScoreResponse])
async def analyze_sites(request: SiteAnalysisRequest):
    """Analyze sites with preset or custom weights.
    
    Returns ranked sites with component breakdowns and explanations.
    """
    try:
        # Load preset if specified
        weights = None
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, "site")
            
            if preset is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset: {request.preset_id}"
                )
            
            weights = preset.get_weights_dict()
        
        # Override with custom weights if provided
        if request.custom_weights:
            if weights is None:
                weights = {}
            weights.update(request.custom_weights)
        
        # Run analysis pipeline
        pipeline = AnalysisPipeline()
        roi = BoundingBox(**request.roi)
        
        results = pipeline.run(
            roi,
            dataset=request.dataset,
            threshold=request.threshold,
            criteria_weights=weights
        )
        
        # Score sites with explainability
        from marshab.config.preset_loader import PresetLoader
        loader = PresetLoader()
        loader.load()
        
        beneficial = {
            name: loader.get_criterion(name).beneficial
            for name in results.criteria.keys()
            if loader.get_criterion(name) is not None
        }
        
        scoring_engine = SiteScoringEngine()
        scored_sites = scoring_engine.score_sites(
            results.sites,
            results.criteria,
            weights or {},
            beneficial
        )
        
        # Format response
        response = []
        for scored_site in scored_sites:
            response.append(SiteScoreResponse(
                site_id=scored_site.site.site_id,
                rank=scored_site.site.rank,
                total_score=scored_site.total_score,
                components=scored_site.components,
                explanation=scored_site.explanation,
                geometry=scored_site.site.geometry.__geo_interface__,
                centroid_lat=scored_site.site.centroid_lat,
                centroid_lon=scored_site.site.centroid_lon,
                area_km2=scored_site.site.area_km2
            ))
        
        return response
        
    except Exception as e:
        logger.error("Site analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Task 3: Route Cost API

### Implementation

**File:** `marshab/analysis/route_cost.py` (new)

```python
"""Route cost analysis with breakdown."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RouteCostBreakdown:
    """Route cost with component breakdown."""
    total_cost: float
    distance_m: float
    slope_penalty: float
    roughness_penalty: float
    elevation_penalty: float
    components: Dict[str, float]
    explanation: str


class RouteCostEngine:
    """Analyzes route costs with explainability."""
    
    def analyze_route(
        self,
        waypoints: pd.DataFrame,
        weights: Dict[str, float]
    ) -> RouteCostBreakdown:
        """Analyze route cost with component breakdown.
        
        Args:
            waypoints: Waypoint DataFrame with x_site, y_site
            weights: Cost component weights
            
        Returns:
            RouteCostBreakdown with explanation
        """
        logger.info("Analyzing route cost")
        
        # Calculate distance
        distances = []
        for i in range(len(waypoints) - 1):
            dx = waypoints.iloc[i+1]['x_site'] - waypoints.iloc[i]['x_site']
            dy = waypoints.iloc[i+1]['y_site'] - waypoints.iloc[i]['y_site']
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
        
        total_distance = sum(distances)
        
        # Placeholder for terrain penalties
        # In real implementation, would query terrain along path
        slope_penalty = 50.0  # Stub
        roughness_penalty = 30.0  # Stub
        elevation_penalty = 20.0  # Stub
        
        # Calculate weighted cost
        components = {
            "distance": total_distance * weights.get("distance", 0.3),
            "slope": slope_penalty * weights.get("slope_penalty", 0.3),
            "roughness": roughness_penalty * weights.get("roughness_penalty", 0.2),
            "elevation": elevation_penalty * weights.get("elevation_penalty", 0.2)
        }
        
        total_cost = sum(components.values())
        
        # Generate explanation
        explanation = self._generate_explanation(
            total_distance, slope_penalty, roughness_penalty
        )
        
        return RouteCostBreakdown(
            total_cost=total_cost,
            distance_m=total_distance,
            slope_penalty=slope_penalty,
            roughness_penalty=roughness_penalty,
            elevation_penalty=elevation_penalty,
            components=components,
            explanation=explanation
        )
    
    def _generate_explanation(
        self,
        distance: float,
        slope_penalty: float,
        roughness_penalty: float
    ) -> str:
        """Generate route explanation."""
        parts = []
        
        parts.append(f"Total distance: {distance:.0f}m")
        
        if slope_penalty > 100:
            parts.append("includes steep terrain sections")
        elif slope_penalty > 50:
            parts.append("has moderate slope challenges")
        else:
            parts.append("traverses gentle slopes")
        
        if roughness_penalty > 50:
            parts.append("rough surface conditions")
        elif roughness_penalty > 20:
            parts.append("moderate surface roughness")
        else:
            parts.append("smooth terrain")
        
        return "This route " + ", ".join(parts) + "."
```

**File:** `marshab/web/routes/route_analysis.py` (new)

```python
"""Route cost analysis API."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict
from pathlib import Path

from marshab.core.navigation_engine import NavigationEngine
from marshab.analysis.route_cost import RouteCostEngine
from marshab.config.preset_loader import PresetLoader
from marshab.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


class RouteCostRequest(BaseModel):
    """Request for route cost analysis."""
    site_id_start: int
    site_id_end: int
    analysis_dir: str
    preset_id: Optional[str] = None
    custom_weights: Optional[Dict[str, float]] = None


class RouteCostResponse(BaseModel):
    """Route cost analysis response."""
    total_cost: float
    distance_m: float
    components: Dict[str, float]
    explanation: str
    num_waypoints: int


@router.post("/route-cost", response_model=RouteCostResponse)
async def analyze_route_cost(request: RouteCostRequest):
    """Analyze route cost with preset or custom weights."""
    try:
        # Load preset
        weights = {}
        if request.preset_id:
            loader = PresetLoader()
            preset = loader.get_preset(request.preset_id, "route")
            
            if preset is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset: {request.preset_id}"
                )
            
            weights = preset.get_weights_dict()
        
        # Override with custom
        if request.custom_weights:
            weights.update(request.custom_weights)
        
        # Get waypoints (assuming already generated)
        waypoints_file = Path(request.analysis_dir) / f"waypoints_{request.site_id_start}_to_{request.site_id_end}.csv"
        
        if not waypoints_file.exists():
            raise HTTPException(
                status_code=404,
                detail="Route waypoints not found. Generate route first."
            )
        
        import pandas as pd
        waypoints = pd.read_csv(waypoints_file)
        
        # Analyze cost
        engine = RouteCostEngine()
        breakdown = engine.analyze_route(waypoints, weights)
        
        return RouteCostResponse(
            total_cost=breakdown.total_cost,
            distance_m=breakdown.distance_m,
            components=breakdown.components,
            explanation=breakdown.explanation,
            num_waypoints=len(waypoints)
        )
        
    except Exception as e:
        logger.error("Route cost analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Task 4: Decision Lab UI

### Implementation

**File:** `webui/src/pages/DecisionLab.tsx` (new)

```typescript
import React, { useState, useEffect } from 'react'
import PresetsSelector from '../components/PresetsSelector'
import AdvancedWeightsPanel from '../components/AdvancedWeightsPanel'
import SiteScoresList from '../components/SiteScoresList'
import TerrainMap from '../components/TerrainMap'
import Terrain3D from '../components/Terrain3D'

interface ROI {
  lat_min: number
  lat_max: number
  lon_min: number
  lon_max: number
}

export default function DecisionLab() {
  const [roi, setROI] = useState<ROI | null>(null)
  const [dataset, setDataset] = useState('mola')
  const [selectedPreset, setSelectedPreset] = useState<string>('balanced')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [customWeights, setCustomWeights] = useState<Record<string, number>>({})
  const [sites, setSites] = useState<any[]>([])
  const [selectedSite, setSelectedSite] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  // Fetch presets on mount
  const [presets, setPresets] = useState<any[]>([])
  
  useEffect(() => {
    fetch('http://localhost:5000/api/v1/analysis/presets')
      .then(res => res.json())
      .then(data => setPresets(data.site_presets))
      .catch(err => console.error('Failed to load presets:', err))
  }, [])

  // Run analysis when ROI or preset changes
  const runAnalysis = async () => {
    if (!roi) return
    
    setLoading(true)
    
    try {
      const request = {
        roi,
        dataset,
        preset_id: selectedPreset,
        custom_weights: Object.keys(customWeights).length > 0 ? customWeights : null,
        threshold: 0.6
      }
      
      const response = await fetch('http://localhost:5000/api/v1/analysis/site-scores', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      })
      
      if (response.ok) {
        const data = await response.json()
        setSites(data)
      } else {
        console.error('Analysis failed:', response.statusText)
      }
    } catch (error) {
      console.error('Analysis error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <h1 className="text-2xl font-bold">Mars Landing Site Decision Lab</h1>
        <p className="text-gray-400 text-sm">
          Explore landing sites using preset criteria or customize your own
        </p>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: Controls */}
        <div className="w-96 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          {/* ROI Selection */}
          <div className="p-4 border-b border-gray-700">
            <h3 className="font-semibold mb-2">Region of Interest</h3>
            <div className="space-y-2 text-sm">
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="number"
                  placeholder="Lat Min"
                  className="bg-gray-700 p-2 rounded"
                  onChange={(e) => setROI(prev => ({ ...prev!, lat_min: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lat Max"
                  className="bg-gray-700 p-2 rounded"
                  onChange={(e) => setROI(prev => ({ ...prev!, lat_max: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lon Min"
                  className="bg-gray-700 p-2 rounded"
                  onChange={(e) => setROI(prev => ({ ...prev!, lon_min: parseFloat(e.target.value) }))}
                />
                <input
                  type="number"
                  placeholder="Lon Max"
                  className="bg-gray-700 p-2 rounded"
                  onChange={(e) => setROI(prev => ({ ...prev!, lon_max: parseFloat(e.target.value) }))}
                />
              </div>
            </div>
          </div>

          {/* Preset Selection */}
          <PresetsSelector
            presets={presets}
            selected={selectedPreset}
            onSelect={setSelectedPreset}
          />

          {/* Advanced Weights */}
          <div className="p-4 border-b border-gray-700">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-blue-400 hover:text-blue-300"
            >
              {showAdvanced ? '▼' : '▶'} Advanced Weights
            </button>
            
            {showAdvanced && (
              <AdvancedWeightsPanel
                weights={customWeights}
                onChange={setCustomWeights}
              />
            )}
          </div>

          {/* Run Analysis Button */}
          <div className="p-4">
            <button
              onClick={runAnalysis}
              disabled={!roi || loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold"
            >
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </div>

          {/* Site Scores List */}
          {sites.length > 0 && (
            <SiteScoresList
              sites={sites}
              selectedSite={selectedSite}
              onSelectSite={setSelectedSite}
            />
          )}
        </div>

        {/* Right Panel: Visualization */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1 p-4">
            {roi && (
              <Terrain3D
                roi={roi}
                dataset={dataset}
                showSites={sites.length > 0}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
```

---

## Deliverables Checklist

- [ ] `criteria_presets.yaml` created with site and route presets
- [ ] `PresetLoader` class implemented and tested
- [ ] Site scoring API with explainability complete
- [ ] Route cost API with breakdown functional
- [ ] Decision Lab UI page implemented
- [ ] Preset selector component created
- [ ] Advanced weights panel with progressive disclosure
- [ ] Site scores list with explanations
- [ ] Integration tests passing
- [ ] Documentation updated

---

## Success Metrics

1. ✅ Users can select presets and see results immediately
2. ✅ Advanced panel hidden by default, accessible on demand
3. ✅ Site explanations are clear and actionable
4. ✅ Custom weights produce different rankings
5. ✅ UI responsive and intuitive
6. ✅ All presets validated and tested
7. ✅ API endpoints return consistent responses
