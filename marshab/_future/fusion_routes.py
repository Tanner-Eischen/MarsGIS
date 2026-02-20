"""API routes for multi-resolution data fusion. Not mounted by default."""

from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.core.data_manager import DataManager
from marshab.models import BoundingBox
from marshab.utils.logging import get_logger

from .multi_resolution_fusion import FusionParameters, MultiResolutionFusion

logger = get_logger(__name__)
router = APIRouter(prefix="/fusion", tags=["fusion"])


class FusionRequest(BaseModel):
    """Request model for multi-resolution data fusion."""
    roi: BoundingBox = Field(..., description="Region of interest bounding box")
    datasets: list[Literal["mola", "mola_200m", "hirise", "ctx"]] = Field(
        ...,
        description="List of datasets to fuse",
        min_items=1,
        max_items=3
    )
    primary_dataset: Literal["mola", "mola_200m", "hirise", "ctx"] = Field(
        "mola",
        description="Primary dataset for fusion"
    )
    blending_method: Literal["weighted_average", "hierarchical", "adaptive"] = Field(
        "weighted_average",
        description="Method for blending datasets"
    )
    upsampling_method: Literal["bilinear", "bicubic", "lanczos", "kriging"] = Field(
        "bilinear",
        description="Method for upsampling lower resolution data"
    )
    downsampling_method: Literal["average", "median", "gaussian", "decimation"] = Field(
        "average",
        description="Method for downsampling higher resolution data"
    )
    target_resolution: Optional[float] = Field(
        None,
        description="Target resolution in meters. If not specified, uses primary dataset resolution."
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for data inclusion"
    )
    edge_preservation: bool = Field(
        True,
        description="Whether to preserve edges during resampling"
    )
    noise_reduction: bool = Field(
        True,
        description="Whether to apply noise reduction during fusion"
    )


class FusionResponse(BaseModel):
    """Response model for multi-resolution data fusion."""
    success: bool
    fused_data: dict = Field(..., description="Fused elevation data")
    fusion_info: dict = Field(..., description="Information about the fusion process")
    quality_metrics: dict = Field(..., description="Quality metrics for the fused dataset")
    dataset_info: dict = Field(..., description="Information about input datasets")


class DatasetInfoResponse(BaseModel):
    """Response model for dataset information."""
    available_datasets: list[str] = Field(..., description="List of available datasets")
    dataset_configs: dict[str, dict] = Field(..., description="Configuration for each dataset")


class FusionMethodsResponse(BaseModel):
    """Response model for fusion methods information."""
    blending_methods: list[str] = Field(..., description="Available blending methods")
    upsampling_methods: list[str] = Field(..., description="Available upsampling methods")
    downsampling_methods: list[str] = Field(..., description="Available downsampling methods")


@router.get("/info", response_model=DatasetInfoResponse)
async def get_fusion_info():
    """Get information about available datasets for fusion."""
    try:
        fusion_service = MultiResolutionFusion(
            FusionParameters(
                primary_dataset="mola",
                blending_method="weighted_average",
                upsampling_method="bilinear",
                downsampling_method="average",
                confidence_threshold=0.5,
                edge_preservation=True,
                noise_reduction=True
            )
        )

        dataset_configs = {
            dataset_id: {
                "name": config.name,
                "resolution": config.resolution,
                "priority": config.priority,
                "noise_level": config.noise_level,
                "effective_range": config.effective_range
            }
            for dataset_id, config in fusion_service.DATASET_CONFIGS.items()
        }

        return DatasetInfoResponse(
            available_datasets=list(fusion_service.DATASET_CONFIGS.keys()),
            dataset_configs=dataset_configs
        )

    except Exception as e:
        logger.error(f"Error getting fusion info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting fusion info: {str(e)}")


@router.get("/methods", response_model=FusionMethodsResponse)
async def get_fusion_methods():
    """Get information about available fusion methods."""
    return FusionMethodsResponse(
        blending_methods=["weighted_average", "hierarchical", "adaptive"],
        upsampling_methods=["bilinear", "bicubic", "lanczos", "kriging"],
        downsampling_methods=["average", "median", "gaussian", "decimation"]
    )


@router.post("/fuse", response_model=FusionResponse)
async def fuse_datasets(request: FusionRequest):
    """Perform multi-resolution data fusion on multiple datasets."""
    try:
        logger.info(f"Starting multi-resolution fusion for datasets: {request.datasets}")

        fusion_params = FusionParameters(
            primary_dataset=request.primary_dataset,
            blending_method=request.blending_method,
            upsampling_method=request.upsampling_method,
            downsampling_method=request.downsampling_method,
            confidence_threshold=request.confidence_threshold,
            edge_preservation=request.edge_preservation,
            noise_reduction=request.noise_reduction
        )

        fusion_service = MultiResolutionFusion(fusion_params)
        data_manager = DataManager()

        for dataset_id in request.datasets:
            try:
                data = data_manager.get_dem_for_roi(
                    request.roi, dataset=dataset_id, download=True, clip=True
                )

                if data is None:
                    continue

                fusion_service.add_dataset(dataset_id, data, request.roi)

            except Exception as e:
                logger.error(f"Error loading dataset {dataset_id}: {e}")
                continue

        if not fusion_service.datasets:
            raise HTTPException(
                status_code=404,
                detail="No datasets could be loaded for the specified ROI"
            )

        fused_dataset = fusion_service.fuse_datasets(request.target_resolution)
        quality_metrics = fusion_service.get_fusion_quality_metrics()
        dataset_info = fusion_service.get_dataset_info()

        fused_data = {
            "elevation": fused_dataset.values.tolist(),
            "lat": fused_dataset.lat.values.tolist(),
            "lon": fused_dataset.lon.values.tolist(),
            "shape": fused_dataset.shape,
            "attrs": dict(fused_dataset.attrs)
        }

        fusion_info = {
            "method": request.blending_method,
            "target_resolution": request.target_resolution or fusion_service.DATASET_CONFIGS[request.primary_dataset].resolution,
            "primary_dataset": request.primary_dataset,
            "input_datasets": list(fusion_service.datasets.keys()),
            "fusion_timestamp": logger.name
        }

        return FusionResponse(
            success=True,
            fused_data=fused_data,
            fusion_info=fusion_info,
            quality_metrics=quality_metrics,
            dataset_info=dataset_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during data fusion: {e}")
        raise HTTPException(status_code=500, detail=f"Error during data fusion: {str(e)}")


@router.post("/compare", response_model=FusionResponse)
async def compare_datasets(request: FusionRequest):
    """Compare multiple datasets without fusion (for analysis)."""
    try:
        data_manager = DataManager()
        datasets_data = {}

        for dataset_id in request.datasets:
            try:
                data = data_manager.get_dem_for_roi(
                    request.roi, dataset=dataset_id, download=True, clip=True
                )

                if data is not None:
                    datasets_data[dataset_id] = {
                        "elevation": data.values.tolist(),
                        "lat": data.lat.values.tolist(),
                        "lon": data.lon.values.tolist(),
                        "shape": data.shape,
                        "stats": {
                            "min": float(data.min()),
                            "max": float(data.max()),
                            "mean": float(data.mean()),
                            "std": float(data.std())
                        }
                    }

            except Exception as e:
                logger.error(f"Error loading dataset {dataset_id} for comparison: {e}")
                continue

        if not datasets_data:
            raise HTTPException(
                status_code=404,
                detail="No datasets could be loaded for comparison"
            )

        comparison_metrics = {}
        if len(datasets_data) >= 2:
            dataset_ids = list(datasets_data.keys())
            for i in range(len(dataset_ids)):
                for j in range(i + 1, len(dataset_ids)):
                    id1, id2 = dataset_ids[i], dataset_ids[j]
                    data1 = np.array(datasets_data[id1]["elevation"])
                    data2 = np.array(datasets_data[id2]["elevation"])
                    if data1.shape == data2.shape:
                        diff = data1 - data2
                        comparison_metrics[f"{id1}_vs_{id2}"] = {
                            "mean_difference": float(np.mean(diff)),
                            "std_difference": float(np.std(diff)),
                            "max_difference": float(np.max(np.abs(diff)))
                        }

        return FusionResponse(
            success=True,
            fused_data=datasets_data,
            fusion_info={
                "comparison_mode": True,
                "datasets_compared": list(datasets_data.keys()),
                "comparison_metrics": comparison_metrics
            },
            quality_metrics={"comparison_completed": True},
            dataset_info={"mode": "comparison"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during dataset comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Error during dataset comparison: {str(e)}")


@router.get("/example")
async def get_fusion_example():
    """Get an example fusion request for testing."""
    return {
        "description": "Example multi-resolution data fusion request",
        "request": {
            "roi": {
                "lat_min": 40.0,
                "lat_max": 41.0,
                "lon_min": 180.0,
                "lon_max": 181.0
            },
            "datasets": ["mola", "ctx", "hirise"],
            "primary_dataset": "mola",
            "blending_method": "weighted_average",
            "upsampling_method": "bilinear",
            "downsampling_method": "average",
            "target_resolution": 100.0,
            "confidence_threshold": 0.5,
            "edge_preservation": True,
            "noise_reduction": True
        },
    }
