"""Unit tests for MCDM evaluation."""

import numpy as np
import pytest

from marshab.processing.mcdm import MCDMEvaluator


class TestMCDMEvaluator:
    """Tests for MCDMEvaluator class."""

    def test_normalize_beneficial(self):
        """Test normalization of beneficial criterion."""
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=True)

        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)
        assert normalized[0] < normalized[4]  # Lower values are lower

    def test_normalize_cost(self):
        """Test normalization of cost criterion."""
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=False)

        assert np.min(normalized) == pytest.approx(0.0)
        assert np.max(normalized) == pytest.approx(1.0)
        assert normalized[0] > normalized[4]  # Higher values are better (inverted)

    def test_normalize_constant_values(self):
        """Test normalization with constant values."""
        data = np.array([5, 5, 5, 5, 5], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=True)

        # Should return 0.5 for all values when min == max
        assert np.allclose(normalized, 0.5)

    def test_normalize_with_nan(self):
        """Test normalization handles NaN values."""
        data = np.array([10, 20, np.nan, 40, 50], dtype=float)
        normalized = MCDMEvaluator.normalize_criterion(data, beneficial=True)

        # NaN should remain NaN
        assert np.isnan(normalized[2])
        # Other values should be normalized
        assert np.min(normalized[~np.isnan(normalized)]) == pytest.approx(0.0)
        assert np.max(normalized[~np.isnan(normalized)]) == pytest.approx(1.0)

    def test_weighted_sum(self):
        """Test weighted sum MCDM."""
        criteria = {
            "slope": np.array([5, 10, 15, 20, 25], dtype=float),
            "roughness": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
        }
        weights = {"slope": 0.6, "roughness": 0.4}
        beneficial = {"slope": False, "roughness": False}

        suitability = MCDMEvaluator.weighted_sum(criteria, weights, beneficial)

        assert suitability.shape == (5,)
        assert np.all(suitability >= 0)
        assert np.all(suitability <= 1)
        assert suitability[0] > suitability[4]  # Lower is better for both

    def test_weighted_sum_invalid_weights(self):
        """Test error handling for invalid weight sums."""
        criteria = {
            "slope": np.array([5, 10, 15], dtype=float),
        }
        weights = {"slope": 0.5}  # Doesn't sum to 1.0
        beneficial = {"slope": False}

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            MCDMEvaluator.weighted_sum(criteria, weights, beneficial)

    def test_weighted_sum_multiple_criteria(self):
        """Test weighted sum with multiple criteria."""
        criteria = {
            "slope": np.array([1, 2, 3], dtype=float),
            "roughness": np.array([0.1, 0.2, 0.3], dtype=float),
            "elevation": np.array([100, 200, 300], dtype=float),
        }
        weights = {"slope": 0.3, "roughness": 0.3, "elevation": 0.4}
        beneficial = {"slope": False, "roughness": False, "elevation": True}

        suitability = MCDMEvaluator.weighted_sum(criteria, weights, beneficial)

        assert suitability.shape == (3,)
        assert np.all(suitability >= 0)
        assert np.all(suitability <= 1)

    def test_topsis(self):
        """Test TOPSIS method."""
        criteria = {
            "slope": np.array([5, 10, 15, 20], dtype=float),
            "roughness": np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        }
        weights = {"slope": 0.6, "roughness": 0.4}
        beneficial = {"slope": False, "roughness": False}

        topsis_score = MCDMEvaluator().topsis(criteria, weights, beneficial)

        assert topsis_score.shape == (4,)
        assert np.all(topsis_score >= 0)
        assert np.all(topsis_score <= 1)

    def test_topsis_with_nan(self):
        """Test TOPSIS handles NaN values."""
        criteria = {
            "slope": np.array([5, 10, np.nan, 20], dtype=float),
            "roughness": np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        }
        weights = {"slope": 0.5, "roughness": 0.5}
        beneficial = {"slope": False, "roughness": False}

        topsis_score = MCDMEvaluator().topsis(criteria, weights, beneficial)

        # Should handle NaN gracefully
        assert topsis_score.shape == (4,)
        # NaN in input may propagate, but should not crash

    def test_topsis_ideal_solution(self):
        """Test TOPSIS with clear ideal solution."""
        # Create criteria where one alternative is clearly best
        criteria = {
            "benefit": np.array([1, 10, 1], dtype=float),  # Middle is best
            "cost": np.array([10, 1, 10], dtype=float),  # Middle is best
        }
        weights = {"benefit": 0.5, "cost": 0.5}
        beneficial = {"benefit": True, "cost": False}

        topsis_score = MCDMEvaluator().topsis(criteria, weights, beneficial)

        # Middle alternative should have highest score
        assert topsis_score[1] > topsis_score[0]
        assert topsis_score[1] > topsis_score[2]

    def test_weighted_sum_edge_case_single_criterion(self):
        """Test weighted sum with single criterion."""
        criteria = {"slope": np.array([1, 2, 3, 4, 5], dtype=float)}
        weights = {"slope": 1.0}
        beneficial = {"slope": False}

        suitability = MCDMEvaluator.weighted_sum(criteria, weights, beneficial)

        assert suitability.shape == (5,)
        assert np.all(suitability >= 0)
        assert np.all(suitability <= 1)
        # Should be monotonically decreasing (lower slope is better)
        assert suitability[0] > suitability[4]

