"""Multi-criteria suitability with explicit common-valid-cell semantics."""

from typing import Literal

import numpy as np


class MCDMEvaluator:
    """Weighted sum and TOPSIS over cells valid in every active criterion.

    Weights must be finite, nonnegative and sum to one. Criteria with zero
    weight do not affect either the score or the common validity mask.
    """

    @staticmethod
    def normalize_criterion(data: np.ndarray, beneficial: bool = True) -> np.ndarray:
        """Min-max normalize finite cells; constant valid data scores 0.5.

        Invalid cells remain NaN, including when all cells are invalid.
        """
        data = np.asarray(data, dtype=np.float64)
        valid = np.isfinite(data)
        normalized = np.full(data.shape, np.nan, dtype=np.float64)
        if not np.any(valid):
            return normalized
        values = data[valid]
        data_min, data_max = values.min(), values.max()
        if data_min == data_max:
            normalized[valid] = 0.5
        elif beneficial:
            normalized[valid] = (values - data_min) / (data_max - data_min)
        else:
            normalized[valid] = (data_max - values) / (data_max - data_min)
        return normalized

    @staticmethod
    def _prepare(
        criteria: dict[str, np.ndarray], weights: dict[str, float]
    ) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
        """Validate inputs and extract active arrays and their joint mask."""
        if not criteria:
            raise ValueError("At least one criterion is required")
        arrays = {name: np.asarray(data, dtype=np.float64) for name, data in criteria.items()}
        shape = next(iter(arrays.values())).shape
        if any(data.shape != shape for data in arrays.values()):
            raise ValueError("Criteria must have identical shapes")
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Weights must be finite and nonnegative")
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, rtol=0.0, atol=1e-8):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        missing = weights.keys() - arrays.keys()
        if missing:
            raise ValueError(f"Missing weighted criteria: {sorted(missing)}")
        active_weights = {
            name: weight / total_weight for name, weight in weights.items() if weight > 0
        }
        active = {name: arrays[name] for name in active_weights}
        valid = np.ones(shape, dtype=bool)
        for data in active.values():
            valid &= np.isfinite(data)
        return active, active_weights, valid

    @staticmethod
    def weighted_sum(
        criteria: dict[str, np.ndarray],
        weights: dict[str, float],
        beneficial: dict[str, bool],
    ) -> np.ndarray:
        """Sum weighted min-max criteria using the joint valid population.

        Each output cell is NaN if any positive-weight criterion is invalid.
        All active criteria are normalized using only jointly valid cells.
        """
        active, active_weights, valid = MCDMEvaluator._prepare(criteria, weights)
        suitability = np.full(valid.shape, np.nan, dtype=np.float64)
        if not np.any(valid):
            return suitability
        scores = np.zeros(np.count_nonzero(valid), dtype=np.float64)
        for name, data in active.items():
            normalized = MCDMEvaluator.normalize_criterion(data[valid], beneficial.get(name, True))
            scores += normalized * active_weights[name]
        suitability[valid] = scores
        return suitability

    @staticmethod
    def topsis(
        criteria: dict[str, np.ndarray],
        weights: dict[str, float],
        beneficial: dict[str, bool],
    ) -> np.ndarray:
        """Vector-normalized TOPSIS on the joint valid population.

        Scores are distances to the worst ideal divided by the sum of the
        distances to both ideals. Coincident ideals score 0.5; invalid cells
        remain NaN. Constant zero criteria contribute zero distance.
        """
        active, active_weights, valid = MCDMEvaluator._prepare(criteria, weights)
        result = np.full(valid.shape, np.nan, dtype=np.float64)
        if not np.any(valid):
            return result
        dist_best_sq = np.zeros(np.count_nonzero(valid), dtype=np.float64)
        dist_worst_sq = np.zeros_like(dist_best_sq)
        for name, data in active.items():
            values = data[valid]
            # Scaling before the norm prevents overflow for large finite values.
            scale = np.max(np.abs(values))
            scaled = values / scale if scale > 0 else values
            norm = np.linalg.norm(scaled)
            weighted = (scaled / norm if norm > 0 else scaled) * active_weights[name]
            best, worst = weighted.max(), weighted.min()
            if not beneficial.get(name, True):
                best, worst = worst, best
            dist_best_sq += (weighted - best) ** 2
            dist_worst_sq += (weighted - worst) ** 2
        dist_best, dist_worst = np.sqrt(dist_best_sq), np.sqrt(dist_worst_sq)
        denominator = dist_best + dist_worst
        scores = np.full(denominator.shape, 0.5)
        np.divide(dist_worst, denominator, out=scores, where=denominator > 0)
        result[valid] = scores
        return result

    @staticmethod
    def evaluate(
        criteria: dict[str, np.ndarray],
        weights: dict[str, float],
        beneficial: dict[str, bool],
        method: Literal["weighted_sum", "topsis"] = "weighted_sum",
    ) -> np.ndarray:
        """Evaluate suitability using the specified MCDM method."""
        if method == "weighted_sum":
            return MCDMEvaluator.weighted_sum(criteria, weights, beneficial)
        if method == "topsis":
            return MCDMEvaluator.topsis(criteria, weights, beneficial)
        raise ValueError(f"Unknown MCDM method: {method}")
