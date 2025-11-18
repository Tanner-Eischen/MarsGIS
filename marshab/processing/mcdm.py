"""Multi-Criteria Decision Making (MCDM) for site suitability."""

from typing import Dict, Literal

import numpy as np

from marshab.utils.logging import get_logger

logger = get_logger(__name__)


class MCDMEvaluator:
    """Multi-criteria decision making using weighted sum and TOPSIS."""

    @staticmethod
    def normalize_criterion(
        data: np.ndarray, beneficial: bool = True
    ) -> np.ndarray:
        """Normalize criterion to 0-1 range.
        
        Args:
            data: Input criterion array
            beneficial: True if higher values are better
        
        Returns:
            Normalized array in [0, 1] range
        """
        # Handle empty arrays
        if data.size == 0:
            logger.warning("Empty array provided to normalize_criterion, returning empty array")
            return data
        
        # Handle arrays with all NaN
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            logger.warning("All values are NaN in normalize_criterion, returning zeros")
            return np.zeros_like(data)
        
        # Calculate min/max only on valid values
        valid_data = data[valid_mask]
        data_min = np.nanmin(valid_data) if valid_data.size > 0 else 0.0
        data_max = np.nanmax(valid_data) if valid_data.size > 0 else 1.0
        
        if data_max == data_min or not np.isfinite(data_max) or not np.isfinite(data_min):
            return np.ones_like(data) * 0.5
        
        if beneficial:
            # Higher is better
            normalized = (data - data_min) / (data_max - data_min)
        else:
            # Lower is better (cost criterion)
            normalized = (data_max - data) / (data_max - data_min)
        
        # Preserve NaN values - don't replace with 0.5
        # Only replace infinite values, keep NaN as NaN
        normalized = np.where(np.isfinite(normalized), normalized, np.nan)
        # Replace inf with NaN
        normalized = np.where(np.isinf(normalized), np.nan, normalized)

        return normalized

    @staticmethod
    def weighted_sum(
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool],
    ) -> np.ndarray:
        """Calculate weighted sum of normalized criteria.
        
        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight
            beneficial: Dictionary of criterion name -> benefit direction
        
        Returns:
            Suitability score array in [0, 1] range
        """
        # Validate weights sum to 1.0
        total_weight = sum(weights.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Normalize all criteria
        normalized = {}
        for name, data in criteria.items():
            is_beneficial = beneficial.get(name, True)
            normalized[name] = MCDMEvaluator.normalize_criterion(data, is_beneficial)

        # Calculate weighted sum
        suitability = np.zeros_like(list(criteria.values())[0], dtype=np.float32)

        for name, norm_data in normalized.items():
            weight = weights.get(name, 0.0)
            suitability += norm_data * weight

            logger.debug(
                f"Applied criterion: {name}",
                weight=weight,
                mean_value=float(np.nanmean(norm_data)),
            )

        # Handle empty suitability array
        if suitability.size > 0:
            logger.info(
                "Weighted sum computed",
                mean_suitability=float(np.nanmean(suitability)),
                max_suitability=float(np.nanmax(suitability)),
            )
        else:
            logger.warning("Empty suitability array computed")
        
        return suitability
    
    @staticmethod
    def topsis(
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool]
    ) -> np.ndarray:
        """TOPSIS (Technique for Order Preference by Similarity to Ideal).
        
        Args:
            criteria: Dictionary of criterion name -> array
            weights: Dictionary of criterion name -> weight
            beneficial: Dictionary of criterion name -> benefit direction
        
        Returns:
            TOPSIS score array in [0, 1] range
        """
        # Vector normalization
        normalized = {}
        for name, data in criteria.items():
            norm = np.sqrt(np.sum(data**2 + 1e-10))
            normalized[name] = data / norm if norm > 0 else data
        
        # Apply weights
        weighted = {}
        for name, norm_data in normalized.items():
            weight = weights.get(name, 0.0)
            weighted[name] = norm_data * weight
        
        # Determine ideal best and worst
        ideal_best = {}
        ideal_worst = {}
        
        for name, w_data in weighted.items():
            is_beneficial = beneficial.get(name, True)

            if is_beneficial:
                ideal_best[name] = np.nanmax(w_data)
                ideal_worst[name] = np.nanmin(w_data)
            else:
                ideal_best[name] = np.nanmin(w_data)
                ideal_worst[name] = np.nanmax(w_data)

        # Calculate distances
        dist_best = np.zeros_like(list(weighted.values())[0], dtype=np.float32)
        dist_worst = np.zeros_like(list(weighted.values())[0], dtype=np.float32)
        
        for name, w_data in weighted.items():
            dist_best += (w_data - ideal_best[name])**2
            dist_worst += (w_data - ideal_worst[name])**2
        
        dist_best = np.sqrt(dist_best)
        dist_worst = np.sqrt(dist_worst)
        
        # Calculate TOPSIS score
        # Avoid division by zero
        denominator = dist_best + dist_worst
        topsis_score = np.where(
            denominator > 0,
            dist_worst / denominator,
            0.5
        )
        
        logger.info(
            "TOPSIS computed",
            mean_score=float(np.nanmean(topsis_score)),
            max_score=float(np.nanmax(topsis_score)),
        )
        
        return topsis_score
    
    @staticmethod
    def evaluate(
        criteria: Dict[str, np.ndarray],
        weights: Dict[str, float],
        beneficial: Dict[str, bool],
        method: Literal["weighted_sum", "topsis"] = "weighted_sum"
    ) -> np.ndarray:
        """Evaluate suitability using specified MCDM method.
        
        Args:
            criteria: Criterion name -> values array
            weights: Criterion name -> weight
            beneficial: Criterion name -> benefit direction
            method: MCDM method to use
        
        Returns:
            Suitability score array [0, 1]
        """
        logger.info(f"Evaluating suitability using {method}")
        
        if method == "weighted_sum":
            return MCDMEvaluator.weighted_sum(criteria, weights, beneficial)
        elif method == "topsis":
            return MCDMEvaluator.topsis(criteria, weights, beneficial)
        else:
            raise ValueError(f"Unknown MCDM method: {method}")

