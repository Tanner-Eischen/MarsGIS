"""Base classes for plugins."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr

from marshab.models import BoundingBox


class DatasetPlugin(ABC):
    """Base class for dataset plugins."""

    @abstractmethod
    def get_datasets(self) -> list[dict[str, Any]]:
        """Return list of available datasets.

        Returns:
            List of dataset dictionaries with keys: id, name, description, resolution_m
        """
        pass

    @abstractmethod
    def load_data(self, roi: BoundingBox, dataset_id: str) -> xr.DataArray:
        """Load data for a given ROI and dataset.

        Args:
            roi: Region of interest
            dataset_id: Dataset identifier

        Returns:
            DataArray with elevation data
        """
        pass


class CriterionPlugin(ABC):
    """Base class for criterion plugins."""

    @abstractmethod
    def get_criteria(self) -> list[dict[str, Any]]:
        """Return list of available criteria.

        Returns:
            List of criterion dictionaries with keys: id, name, description, beneficial, unit
        """
        pass

    @abstractmethod
    def calculate(self, dem: xr.DataArray, metrics: dict[str, np.ndarray]) -> np.ndarray:
        """Calculate criterion values.

        Args:
            dem: DEM DataArray
            metrics: Dictionary of terrain metrics (slope, roughness, etc.)

        Returns:
            Criterion values array
        """
        pass

