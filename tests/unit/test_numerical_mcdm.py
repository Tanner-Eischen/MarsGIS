"""Numerical MCDM assertions with dimensionless scores and explicit masks."""

import numpy as np
import pytest

from marshab.processing.mcdm import MCDMEvaluator

METHODS = [MCDMEvaluator.weighted_sum, MCDMEvaluator.topsis]


@pytest.mark.parametrize("beneficial, expected", [(True, [0, 0.5, 1]), (False, [1, 0.5, 0])])
def test_minmax_known_values(beneficial, expected):
    actual = MCDMEvaluator.normalize_criterion(np.array([10, 20, 30]), beneficial)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_constant_and_all_invalid_preserve_nodata():
    actual = MCDMEvaluator.normalize_criterion(np.array([5, np.nan, 5, np.inf]))
    np.testing.assert_allclose(actual, [0.5, np.nan, 0.5, np.nan], equal_nan=True)
    assert np.all(np.isnan(MCDMEvaluator.normalize_criterion(np.array([np.nan, np.inf]))))


def test_weighted_sum_hand_calculation_and_weight_sensitivity():
    criteria = {"safe": np.array([0, 5, 10]), "access": np.array([0, 5, 10])}
    direction = {"safe": False, "access": True}
    low_access = MCDMEvaluator.weighted_sum(criteria, {"safe": 0.75, "access": 0.25}, direction)
    high_access = MCDMEvaluator.weighted_sum(criteria, {"safe": 0.25, "access": 0.75}, direction)
    np.testing.assert_allclose(low_access, [0.75, 0.5, 0.25], atol=1e-12)
    np.testing.assert_allclose(high_access, [0.25, 0.5, 0.75], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_joint_mask_excludes_invalid_alternatives_from_normalization(method):
    # The outlier 1000 must not define an ideal/range: its other criterion is missing.
    criteria = {"a": np.array([0, 10, 1000, np.nan]), "b": np.array([0, 10, np.nan, 10])}
    actual = method(criteria, {"a": 0.5, "b": 0.5}, {"a": True, "b": True})
    np.testing.assert_allclose(actual, [0, 1, np.nan, np.nan], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("method", METHODS)
def test_zero_weight_criterion_cannot_poison_valid_cells(method):
    actual = method(
        {"a": np.array([0, 10]), "unused": np.array([np.nan, np.inf])},
        {"a": 1.0, "unused": 0.0},
        {"a": True},
    )
    np.testing.assert_allclose(actual, [0, 1], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_constant_jointly_valid_population_and_all_invalid(method):
    criteria = {"a": np.array([1, 1, np.nan]), "b": np.array([0, 0, 0])}
    actual = method(criteria, {"a": 0.5, "b": 0.5}, {})
    np.testing.assert_allclose(actual, [0.5, 0.5, np.nan], equal_nan=True)
    all_invalid = method({"a": np.array([np.nan, np.inf])}, {"a": 1}, {})
    assert np.all(np.isnan(all_invalid))


def test_topsis_independent_two_criterion_reference():
    # Both norms are 5; weighted alternatives are (0.3,0.4), (0.4,0.3), (0,0).
    # Benefit a and cost b give best=(0.4,0), worst=(0,0.4).
    actual = MCDMEvaluator.topsis(
        {"a": np.array([3, 4, 0]), "b": np.array([4, 3, 0])},
        {"a": 0.5, "b": 0.5},
        {"a": True, "b": False},
    )
    expected = [0.3 / (0.3 + np.hypot(0.1, 0.4)), np.hypot(0.4, 0.1) / (0.3 + np.hypot(0.4, 0.1)), 0.5]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", [{"a": -1, "b": 2}, {"a": np.nan}, {"a": np.inf}, {"a": 0}, {"a": 0.9}])
def test_invalid_weights_rejected(method, weights):
    with pytest.raises(ValueError, match="Weights"):
        method({"a": np.array([1, 2]), "b": np.array([1, 2])}, weights, {})


@pytest.mark.parametrize("method", METHODS)
def test_missing_weighted_criterion_rejected(method):
    with pytest.raises(ValueError, match="Missing weighted criteria"):
        method({"a": np.array([1, 2])}, {"a": 0.5, "missing": 0.5}, {})


@pytest.mark.parametrize("method", METHODS)
def test_mismatched_shapes_rejected_instead_of_broadcasting(method):
    with pytest.raises(ValueError, match="identical shapes"):
        method({"a": np.zeros((2, 2)), "b": np.zeros((2, 1))}, {"a": 0.5, "b": 0.5}, {})


@pytest.mark.parametrize("method", METHODS)
def test_empty_criteria_rejected(method):
    with pytest.raises(ValueError, match="At least one criterion"):
        method({}, {}, {})
