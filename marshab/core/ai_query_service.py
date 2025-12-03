"""AI-powered natural language query service for Mars site selection."""

import json
import re
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """Result from AI query processing."""
    success: bool
    criteria_weights: Optional[Dict[str, float]] = None
    roi: Optional[Dict[str, float]] = None
    dataset: Optional[str] = None
    explanation: Optional[str] = None
    confidence: float = 0.0
    original_query: str = ""


class AIQueryService:
    """Service for processing natural language queries about Mars site selection."""
    
    # Mapping of natural language terms to criteria
    CRITERIA_KEYWORDS = {
        'slope': ['flat', 'gentle', 'smooth', 'level', 'low slope', 'minimal slope', 'even'],
        'roughness': ['smooth', 'even', 'flat', 'uniform', 'consistent', 'regular'],
        'elevation': ['high', 'low', 'elevated', 'depressed', 'altitude', 'height'],
        'solar_exposure': ['sunny', 'sunlight', 'solar', 'exposed', 'bright', 'illuminated'],
        'resources': ['water', 'ice', 'minerals', 'resources', 'materials', 'subsurface']
    }
    
    # Dataset preferences
    DATASET_KEYWORDS = {
        'mola': ['global', 'overview', 'large scale', 'broad', 'general', 'coarse'],
        'hirise': ['detailed', 'high resolution', 'fine', 'precise', 'close up', 'zoom'],
        'ctx': ['context', 'medium', 'regional', 'area', 'surrounding']
    }
    
    # Location keywords for ROI extraction
    LOCATION_PATTERNS = [
        r'\b(\d+(?:\.\d+)?)[°\s]*([NS])\s*(\d+(?:\.\d+)?)[°\s]*([EW])\b',  # 40°N 180°E
        r'\b(\d+(?:\.\d+)?)\s*([NS])\s*(\d+(?:\.\d+)?)\s*([EW])\b',        # 40 N 180 E
        r'\blatitude\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([NS])?',
        r'\blongitude\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([EW])?',
        r'\blat\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([NS])?',
        r'\blon\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*([EW])?',
    ]
    
    def __init__(self):
        """Initialize the AI query service."""
        logger.info("Initialized AIQueryService")
    
    def process_query(self, query: str) -> QueryResult:
        """Process a natural language query about Mars site selection.
        
        Args:
            query: Natural language query (e.g., "Find me a flat site near water ice deposits")
            
        Returns:
            QueryResult with extracted parameters and confidence score
        """
        try:
            logger.info("Processing AI query", query=query)
            
            # Normalize query
            normalized_query = query.lower().strip()
            
            # Extract criteria weights
            criteria_weights = self._extract_criteria_weights(normalized_query)
            
            # Extract ROI if mentioned
            roi = self._extract_roi(normalized_query)
            
            # Extract dataset preference
            dataset = self._extract_dataset_preference(normalized_query)
            
            # Generate explanation
            explanation = self._generate_explanation(criteria_weights, roi, dataset, normalized_query)
            
            # Calculate confidence based on extraction success
            confidence = self._calculate_confidence(criteria_weights, roi, dataset, normalized_query)
            
            result = QueryResult(
                success=True,
                criteria_weights=criteria_weights,
                roi=roi,
                dataset=dataset,
                explanation=explanation,
                confidence=confidence,
                original_query=query
            )
            
            logger.info("AI query processed successfully", 
                       confidence=confidence, 
                       criteria_weights=criteria_weights,
                       roi=roi,
                       dataset=dataset)
            
            return result
            
        except Exception as e:
            logger.error("Failed to process AI query", error=str(e), query=query)
            return QueryResult(
                success=False,
                explanation=f"Failed to process query: {str(e)}",
                confidence=0.0,
                original_query=query
            )
    
    def _extract_criteria_weights(self, query: str) -> Dict[str, float]:
        """Extract criteria weights from natural language query.
        
        Args:
            query: Normalized query string
            
        Returns:
            Dictionary of criteria weights
        """
        weights = {}
        total_weight = 0.0
        
        # Analyze each criterion
        for criterion, keywords in self.CRITERIA_KEYWORDS.items():
            score = self._calculate_keyword_score(query, keywords)
            if score > 0:
                weights[criterion] = score
                total_weight += score
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Apply importance modifiers
        weights = self._apply_importance_modifiers(query, weights)
        
        return weights
    
    def _calculate_keyword_score(self, query: str, keywords: List[str]) -> float:
        """Calculate relevance score for a set of keywords.
        
        Args:
            query: Query string
            keywords: List of relevant keywords
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        query_words = query.split()
        
        for keyword in keywords:
            # Exact matches get higher scores
            if keyword in query:
                score += 1.0
            # Partial matches get lower scores
            elif any(keyword in word for word in query_words):
                score += 0.5
        
        # Normalize by number of keywords
        if len(keywords) > 0:
            score = min(1.0, score / len(keywords))
        
        return score
    
    def _apply_importance_modifiers(self, query: str, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply importance modifiers based on emphasis words.
        
        Args:
            query: Query string
            weights: Current weights
            
        Returns:
            Modified weights
        """
        # Words that indicate high importance
        high_importance = ['very', 'extremely', 'critically', 'importantly', 'must', 'essential']
        # Words that indicate low importance  
        low_importance = ['somewhat', 'slightly', 'moderately', 'fairly', 'reasonably']
        
        # Check for emphasis patterns
        high_emphasis = any(word in query for word in high_importance)
        low_emphasis = any(word in query for word in low_importance)
        
        if high_emphasis:
            # Boost all weights by 20% if high emphasis detected
            weights = {k: min(1.0, v * 1.2) for k, v in weights.items()}
        elif low_emphasis:
            # Reduce all weights by 20% if low emphasis detected
            weights = {k: max(0.1, v * 0.8) for k, v in weights.items()}
        
        # Re-normalize to ensure they still sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _extract_roi(self, query: str) -> Optional[Dict[str, float]]:
        """Extract region of interest from query.
        
        Args:
            query: Query string
            
        Returns:
            ROI dictionary with lat/lon bounds, or None if not found
        """
        # Look for coordinate patterns
        for pattern in self.LOCATION_PATTERNS:
            match = re.search(pattern, query)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) >= 2:
                        lat = float(groups[0])
                        lon = float(groups[1])
                        
                        # Handle hemisphere indicators
                        if len(groups) >= 4:
                            lat_dir = groups[1] if len(groups) == 4 else 'N'
                            lon_dir = groups[3] if len(groups) == 4 else 'E'
                            
                            if lat_dir.upper() == 'S':
                                lat = -lat
                            if lon_dir.upper() == 'W':
                                lon = -lon
                        
                        # Create a reasonable ROI around the point (1 degree square)
                        roi = {
                            'lat_min': lat - 0.5,
                            'lat_max': lat + 0.5,
                            'lon_min': lon - 0.5,
                            'lon_max': lon + 0.5
                        }
                        
                        # Validate coordinates
                        if self._validate_roi(roi):
                            return roi
                        
                except (ValueError, IndexError):
                    continue
        
        # Look for named regions (basic implementation)
        named_regions = {
            'olympus mons': {'lat_min': 15.0, 'lat_max': 20.0, 'lon_min': 225.0, 'lon_max': 230.0},
            'valles marineris': {'lat_min': -15.0, 'lat_max': 0.0, 'lon_min': 255.0, 'lon_max': 315.0},
            'hellas basin': {'lat_min': -45.0, 'lat_max': -30.0, 'lon_min': 255.0, 'lon_max': 285.0},
            'gale crater': {'lat_min': -5.5, 'lat_max': -4.5, 'lon_min': 137.0, 'lon_max': 138.0},
            'jezero crater': {'lat_min': 18.0, 'lat_max': 19.0, 'lon_min': 77.0, 'lon_max': 78.0},
        }
        
        for region_name, roi in named_regions.items():
            if region_name in query:
                return roi
        
        return None
    
    def _validate_roi(self, roi: Dict[str, float]) -> bool:
        """Validate ROI coordinates.
        
        Args:
            roi: ROI dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            lat_min, lat_max = roi['lat_min'], roi['lat_max']
            lon_min, lon_max = roi['lon_min'], roi['lon_max']
            
            # Basic coordinate validation
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                return False
            if not (0 <= lon_min <= 360 and 0 <= lon_max <= 360):
                return False
            if lat_min >= lat_max:
                return False
            if lon_min >= lon_max:
                return False
            
            # Check reasonable size (max 10 degrees)
            if lat_max - lat_min > 10 or lon_max - lon_min > 10:
                return False
            
            return True
        except (KeyError, TypeError):
            return False
    
    def _extract_dataset_preference(self, query: str) -> Optional[str]:
        """Extract dataset preference from query.
        
        Args:
            query: Query string
            
        Returns:
            Preferred dataset name, or None if not specified
        """
        # Check for explicit dataset mentions
        dataset_scores = {}
        
        for dataset, keywords in self.DATASET_KEYWORDS.items():
            score = self._calculate_keyword_score(query, keywords)
            if score > 0:
                dataset_scores[dataset] = score
        
        # Return the dataset with highest score
        if dataset_scores:
            return max(dataset_scores.keys(), key=lambda k: dataset_scores[k])
        
        return None
    
    def _generate_explanation(self, 
                            criteria_weights: Dict[str, float], 
                            roi: Optional[Dict[str, float]], 
                            dataset: Optional[str], 
                            query: str) -> str:
        """Generate human-readable explanation of the extracted parameters.
        
        Args:
            criteria_weights: Extracted criteria weights
            roi: Extracted ROI
            dataset: Extracted dataset preference
            query: Original query
            
        Returns:
            Explanation string
        """
        parts = []
        
        if criteria_weights:
            criteria_parts = []
            for criterion, weight in sorted(criteria_weights.items(), key=lambda x: x[1], reverse=True):
                percentage = weight * 100
                if percentage > 5:  # Only show significant weights
                    criteria_parts.append(f"{criterion.replace('_', ' ').title()}: {percentage:.0f}%")
            
            if criteria_parts:
                parts.append(f"Site selection will prioritize: {', '.join(criteria_parts)}")
        
        if roi:
            lat_center = (roi['lat_min'] + roi['lat_max']) / 2
            lon_center = (roi['lon_min'] + roi['lon_max']) / 2
            parts.append(f"Search area centered at {lat_center:.1f}°N, {lon_center:.1f}°E")
        
        if dataset:
            dataset_name = dataset.upper()
            resolution = {"MOLA": "463m", "HIRISE": "1m", "CTX": "18m"}.get(dataset_name, "unknown")
            parts.append(f"Using {dataset_name} dataset ({resolution} resolution)")
        
        if not parts:
            return "No specific parameters could be extracted from your query."
        
        return " ".join(parts)
    
    def _calculate_confidence(self, 
                            criteria_weights: Dict[str, float], 
                            roi: Optional[Dict[str, float]], 
                            dataset: Optional[str], 
                            query: str) -> float:
        """Calculate confidence score for the extraction.
        
        Args:
            criteria_weights: Extracted criteria weights
            roi: Extracted ROI
            dataset: Extracted dataset preference
            query: Original query
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Criteria extraction confidence
        if criteria_weights:
            # Higher confidence if we found multiple criteria
            criteria_confidence = min(1.0, len(criteria_weights) * 0.25)
            confidence += criteria_confidence
        
        # ROI extraction confidence
        if roi:
            confidence += 0.3
        
        # Dataset extraction confidence
        if dataset:
            confidence += 0.2
        
        # Query length and clarity
        if len(query.split()) > 3:
            confidence += 0.1
        
        # Cap at 1.0
        return min(1.0, confidence)


# Global instance
ai_query_service = AIQueryService()