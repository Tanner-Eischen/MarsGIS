"""Site scoring with MCDM and explainability."""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

from marshab.processing.mcdm import MCDMEvaluator
from marshab.types import SiteCandidate
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SiteScore:
    """Site score with component breakdown."""
    site: SiteCandidate
    total_score: float
    components: Dict[str, float]  # Criterion name -> contribution
    explanation: str


class SiteScoringEngine:
    """Scores sites using MCDM with explainability."""
    
    def score_sites(
        self,
        sites: List[SiteCandidate],
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> List[SiteScore]:
        """Score sites and provide component breakdowns.
        
        Args:
            sites: List of candidate sites
            criteria: Criterion arrays (full raster arrays)
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
            # For now, we'll use the site's mean values from SiteCandidate
            # In a full implementation, we'd use site geometry to mask the criteria arrays
            site_components = {}
            
            # Map site properties to criteria
            if "slope" in criteria:
                site_components["slope"] = site.mean_slope_deg
            if "roughness" in criteria:
                site_components["roughness"] = site.mean_roughness
            if "elevation" in criteria:
                site_components["elevation"] = site.mean_elevation_m
            if "solar_exposure" in criteria:
                # Estimate from site properties (simplified)
                # In real implementation, would extract from criteria array
                site_components["solar_exposure"] = 0.5  # Placeholder
            if "science_value" in criteria:
                # Placeholder - would come from external data
                site_components["science_value"] = 0.5  # Placeholder
            
            # Use the site's suitability score as total_score
            total_score = site.suitability_score
            
            # Generate explanation
            explanation = self._generate_explanation(
                site_components, weights, beneficial
            )
            
            scored_sites.append(SiteScore(
                site=site,
                total_score=total_score,
                components=site_components,
                explanation=explanation
            ))
        
        # Sort by total score (should already be sorted, but ensure)
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
        
        # Categorize criteria by importance (weight)
        sorted_criteria = sorted(
            components.items(),
            key=lambda x: weights.get(x[0], 0),
            reverse=True
        )
        
        # Focus on top 3 criteria
        for criterion, value in sorted_criteria[:3]:
            weight = weights.get(criterion, 0)
            
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
            
            elif criterion == "elevation":
                if value < -2000:
                    explanations.append("low elevation (good for landing)")
                elif value < 0:
                    explanations.append("moderate elevation")
                else:
                    explanations.append("higher elevation")
            
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
                else:
                    explanations.append("limited scientific value")
        
        if not explanations:
            return "This site meets basic criteria."
        
        return "This site has " + ", ".join(explanations) + "."

