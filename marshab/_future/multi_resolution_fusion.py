"""Multi-resolution data fusion for combining satellite imagery datasets."""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import xarray as xr
from scipy import ndimage

from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a satellite dataset."""
    name: str
    resolution: float  # meters per pixel
    priority: int  # higher priority datasets are preferred in overlap regions
    noise_level: float  # estimated noise standard deviation
    effective_range: tuple[float, float]  # min and max useful elevation values


@dataclass
class FusionParameters:
    """Parameters for multi-resolution data fusion."""
    primary_dataset: str
    blending_method: Literal['weighted_average', 'hierarchical', 'adaptive', 'bayesian']
    upsampling_method: Literal['bilinear', 'bicubic', 'lanczos', 'kriging']
    downsampling_method: Literal['average', 'median', 'gaussian', 'decimation']
    confidence_threshold: float
    edge_preservation: bool
    noise_reduction: bool


class MultiResolutionFusion:
    """Multi-resolution data fusion for satellite imagery datasets."""

    # Dataset configurations for Mars datasets
    DATASET_CONFIGS = {
        'mola': DatasetConfig(
            name='MOLA',
            resolution=463.0,  # meters
            priority=1,
            noise_level=2.0,
            effective_range=(-8000.0, 21000.0)
        ),
        'mola_200m': DatasetConfig(
            name='MOLA 200m',
            resolution=200.0,  # meters
            priority=2,
            noise_level=1.5,
            effective_range=(-8000.0, 21000.0)
        ),
        'hirise': DatasetConfig(
            name='HiRISE',
            resolution=1.0,  # meters
            priority=5,
            noise_level=0.5,
            effective_range=(-2000.0, 5000.0)
        ),
        'ctx': DatasetConfig(
            name='CTX',
            resolution=18.0,  # meters
            priority=3,
            noise_level=1.0,
            effective_range=(-3000.0, 8000.0)
        )
    }

    def __init__(self, parameters: FusionParameters):
        """Initialize the fusion service.

        Args:
            parameters: Fusion parameters and configuration
        """
        self.parameters = parameters
        self.datasets: dict[str, xr.DataArray] = {}
        self.fused_dataset: Optional[xr.DataArray] = None
        self.confidence_map: Optional[np.ndarray] = None
        self.resolution_map: Optional[np.ndarray] = None

    def add_dataset(self, dataset_id: str, data: xr.DataArray, bbox: BoundingBox) -> None:
        """Add a dataset to the fusion process."""
        if dataset_id not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_id}")

        config = self.DATASET_CONFIGS[dataset_id]
        data_min, data_max = float(data.min()), float(data.max())
        range_min, range_max = config.effective_range

        if data_min < range_min or data_max > range_max:
            logger.warning(f"Dataset {dataset_id} contains values outside effective range: "
                         f"[{data_min:.1f}, {data_max:.1f}] vs [{range_min:.1f}, {range_max:.1f}]")

        data.attrs.update({
            'dataset_id': dataset_id,
            'resolution': config.resolution,
            'priority': config.priority,
            'noise_level': config.noise_level,
            'bbox': bbox,
            'config': config
        })

        self.datasets[dataset_id] = data
        logger.info(f"Added dataset {dataset_id} with resolution {config.resolution}m "
                   f"and dimensions {data.shape}")

    def _resample_to_common_grid(self, target_resolution: float) -> dict[str, xr.DataArray]:
        resampled_datasets = {}
        for dataset_id, data in self.datasets.items():
            config = self.DATASET_CONFIGS[dataset_id]
            current_resolution = config.resolution

            if current_resolution == target_resolution:
                resampled_datasets[dataset_id] = data
            elif current_resolution > target_resolution:
                scale_factor = current_resolution / target_resolution
                resampled = self._downsample_dataset(data, scale_factor)
                resampled_datasets[dataset_id] = resampled
            else:
                scale_factor = target_resolution / current_resolution
                resampled = self._upsample_dataset(data, scale_factor)
                resampled_datasets[dataset_id] = resampled

        return resampled_datasets

    def _downsample_dataset(self, data: xr.DataArray, scale_factor: float) -> xr.DataArray:
        method = self.parameters.downsampling_method
        block_size = int(scale_factor)

        if method == 'average':
            if block_size > 1:
                new_shape = (data.shape[0] // block_size, block_size,
                           data.shape[1] // block_size, block_size)
                reshaped = data.values.reshape(new_shape)
                downsampled = np.mean(reshaped, axis=(1, 3))
            else:
                downsampled = data.values
        elif method == 'median':
            if block_size > 1:
                downsampled = ndimage.median_filter(data.values, size=block_size)[::block_size, ::block_size]
            else:
                downsampled = data.values
        elif method == 'gaussian':
            sigma = scale_factor / 2.0
            filtered = ndimage.gaussian_filter(data.values, sigma=sigma)
            downsampled = filtered[::block_size, ::block_size]
        elif method == 'decimation':
            downsampled = data.values[::block_size, ::block_size]
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        new_lat = data.lat.values[::block_size]
        new_lon = data.lon.values[::block_size]

        return xr.DataArray(
            downsampled,
            coords={'lat': new_lat, 'lon': new_lon},
            dims=['lat', 'lon'],
            attrs=data.attrs
        )

    def _upsample_dataset(self, data: xr.DataArray, scale_factor: float) -> xr.DataArray:
        method = self.parameters.upsampling_method

        if method == 'bilinear':
            upsampled = ndimage.zoom(data.values, scale_factor, order=1)
        elif method == 'bicubic':
            upsampled = ndimage.zoom(data.values, scale_factor, order=3)
        elif method == 'lanczos':
            upsampled = ndimage.zoom(data.values, scale_factor, order=5)
        elif method == 'kriging':
            upsampled = self._kriging_upsample(data.values, scale_factor)
        else:
            raise ValueError(f"Unknown upsampling method: {method}")

        new_size_lat = int(data.shape[0] * scale_factor)
        new_size_lon = int(data.shape[1] * scale_factor)
        new_lat = np.linspace(data.lat.values[0], data.lat.values[-1], new_size_lat)
        new_lon = np.linspace(data.lon.values[0], data.lon.values[-1], new_size_lon)

        return xr.DataArray(
            upsampled,
            coords={'lat': new_lat, 'lon': new_lon},
            dims=['lat', 'lon'],
            attrs=data.attrs
        )

    def _kriging_upsample(self, data: np.ndarray, scale_factor: float) -> np.ndarray:
        kernel_size = 5
        sigma = 1.0
        x = np.arange(kernel_size) - kernel_size // 2
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        kernel = kernel[:, np.newaxis] * kernel[np.newaxis, :]
        kernel /= kernel.sum()
        smoothed = ndimage.convolve(data, kernel, mode='reflect')
        return ndimage.zoom(smoothed, scale_factor, order=3)

    def _calculate_confidence_weights(self, datasets: dict[str, xr.DataArray]) -> dict[str, np.ndarray]:
        weights = {}
        for dataset_id, data in datasets.items():
            config = self.DATASET_CONFIGS[dataset_id]
            base_confidence = config.priority / (1.0 + config.noise_level)
            confidence = np.full(data.shape, base_confidence)

            edge_buffer = min(5, min(data.shape) // 10)
            if edge_buffer > 0:
                mask = np.ones(data.shape)
                mask[:edge_buffer, :] = 0
                mask[-edge_buffer:, :] = 0
                mask[:, :edge_buffer] = 0
                mask[:, -edge_buffer:] = 0
                distances = ndimage.distance_transform_edt(mask)
                edge_factor = np.minimum(1.0, distances / edge_buffer)
                confidence *= edge_factor

            data_values = data.values
            valid_mask = ~np.isnan(data_values) & ~np.isinf(data_values)
            if np.any(valid_mask):
                data_min, data_max = np.percentile(data_values[valid_mask], [5, 95])
                range_factor = np.where(
                    (data_values < data_min) | (data_values > data_max), 0.7, 1.0
                )
                confidence *= range_factor

            if self.parameters.noise_reduction and np.any(valid_mask):
                local_variance = ndimage.uniform_filter(data_values**2, size=3) - \
                               (ndimage.uniform_filter(data_values, size=3))**2
                noise_threshold = np.percentile(local_variance[valid_mask], 75)
                noise_factor = np.where(local_variance > noise_threshold, 0.8, 1.0)
                confidence *= noise_factor

            weights[dataset_id] = confidence

        return weights

    def _weighted_average_fusion(self, datasets: dict[str, xr.DataArray],
                                weights: dict[str, np.ndarray]) -> xr.DataArray:
        dataset_ids = list(datasets.keys())
        data_stack = np.stack([datasets[did].values for did in dataset_ids])
        weight_stack = np.stack([weights[did] for did in dataset_ids])
        weight_sum = np.sum(weight_stack, axis=0)
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
        weight_stack_normalized = weight_stack / weight_sum
        fused_data = np.sum(data_stack * weight_stack_normalized, axis=0)
        primary_dataset = datasets[self.parameters.primary_dataset]
        return xr.DataArray(
            fused_data,
            coords=primary_dataset.coords,
            dims=primary_dataset.dims,
            attrs={
                'fusion_method': 'weighted_average',
                'input_datasets': dataset_ids,
                'primary_dataset': self.parameters.primary_dataset
            }
        )

    def _hierarchical_fusion(self, datasets: dict[str, xr.DataArray]) -> xr.DataArray:
        sorted_datasets = sorted(
            datasets.items(),
            key=lambda x: self.DATASET_CONFIGS[x[0]].resolution
        )
        fused_data = sorted_datasets[0][1].values.copy()
        for dataset_id, data in sorted_datasets[1:]:
            residual = data.values - fused_data
            if self.parameters.noise_reduction:
                residual = ndimage.gaussian_filter(residual, sigma=1.0)
            fused_data = fused_data + residual
        primary_dataset = datasets[self.parameters.primary_dataset]
        return xr.DataArray(
            fused_data,
            coords=primary_dataset.coords,
            dims=primary_dataset.dims,
            attrs={
                'fusion_method': 'hierarchical',
                'input_datasets': [did for did, _ in sorted_datasets],
                'primary_dataset': self.parameters.primary_dataset
            }
        )

    def _adaptive_fusion(self, datasets: dict[str, xr.DataArray],
                        weights: dict[str, np.ndarray]) -> xr.DataArray:
        dataset_ids = list(datasets.keys())
        local_variances = {}
        for dataset_id, data in datasets.items():
            data_values = data.values
            local_mean = ndimage.uniform_filter(data_values, size=5)
            local_var = ndimage.uniform_filter(data_values**2, size=5) - local_mean**2
            local_variances[dataset_id] = local_var

        adaptive_weights = {}
        for dataset_id in dataset_ids:
            base_weight = weights[dataset_id]
            variance = np.where(local_variances[dataset_id] == 0, 1e-10, local_variances[dataset_id])
            adaptive_weights[dataset_id] = base_weight / (1.0 + variance)

        weight_sum = np.sum(list(adaptive_weights.values()), axis=0)
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
        for dataset_id in dataset_ids:
            adaptive_weights[dataset_id] = adaptive_weights[dataset_id] / weight_sum

        data_stack = np.stack([datasets[did].values for did in dataset_ids])
        weight_stack = np.stack([adaptive_weights[did] for did in dataset_ids])
        fused_data = np.sum(data_stack * weight_stack, axis=0)
        primary_dataset = datasets[self.parameters.primary_dataset]
        return xr.DataArray(
            fused_data,
            coords=primary_dataset.coords,
            dims=primary_dataset.dims,
            attrs={
                'fusion_method': 'adaptive',
                'input_datasets': dataset_ids,
                'primary_dataset': self.parameters.primary_dataset
            }
        )

    def fuse_datasets(self, target_resolution: Optional[float] = None) -> xr.DataArray:
        if not self.datasets:
            raise ValueError("No datasets available for fusion")
        if len(self.datasets) == 1:
            return list(self.datasets.values())[0]

        if target_resolution is None:
            target_resolution = self.DATASET_CONFIGS[self.parameters.primary_dataset].resolution

        resampled_datasets = self._resample_to_common_grid(target_resolution)
        confidence_weights = self._calculate_confidence_weights(resampled_datasets)

        if self.parameters.blending_method == 'weighted_average':
            fused_dataset = self._weighted_average_fusion(resampled_datasets, confidence_weights)
        elif self.parameters.blending_method == 'hierarchical':
            fused_dataset = self._hierarchical_fusion(resampled_datasets)
        elif self.parameters.blending_method == 'adaptive':
            fused_dataset = self._adaptive_fusion(resampled_datasets, confidence_weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.parameters.blending_method}")

        self.fused_dataset = fused_dataset
        return fused_dataset

    def get_fusion_quality_metrics(self) -> dict[str, float]:
        if self.fused_dataset is None:
            raise ValueError("No fused dataset available")
        fused_data = self.fused_dataset.values
        metrics = {
            'mean_elevation': float(np.nanmean(fused_data)),
            'std_elevation': float(np.nanstd(fused_data)),
            'elevation_range': float(np.nanmax(fused_data) - np.nanmin(fused_data)),
            'data_coverage': float(np.sum(~np.isnan(fused_data)) / fused_data.size),
            'roughness': float(np.nanstd(fused_data)),
        }
        if len(self.datasets) > 1:
            input_data = []
            for dataset_id, data in self.datasets.items():
                if dataset_id != self.parameters.primary_dataset:
                    primary_data = self.datasets[self.parameters.primary_dataset]
                    if data.shape == primary_data.shape:
                        input_data.append(data.values)
            if input_data:
                metrics['inter_dataset_variance'] = float(np.nanmean(np.nanvar(np.stack(input_data), axis=0)))
        return metrics

    def get_dataset_info(self) -> dict[str, dict]:
        info = {}
        for dataset_id, data in self.datasets.items():
            config = self.DATASET_CONFIGS[dataset_id]
            info[dataset_id] = {
                'resolution': config.resolution,
                'priority': config.priority,
                'noise_level': config.noise_level,
                'effective_range': config.effective_range,
                'dimensions': data.shape,
                'elevation_range': [float(data.min()), float(data.max())],
                'coverage': float(np.sum(~np.isnan(data.values)) / data.size)
            }
        return info
