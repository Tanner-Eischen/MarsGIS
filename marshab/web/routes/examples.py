"""Example ROI endpoints."""

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/examples", tags=["examples"])


class ExampleROI(BaseModel):
    """Example ROI response."""
    id: str
    name: str
    description: str
    bbox: dict
    dataset: str


@router.get("/rois", response_model=list[ExampleROI])
async def get_example_rois():
    """Get list of example ROIs with descriptions."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "example_rois.yaml"

        if not config_path.exists():
            logger.warning("Example ROIs config not found", path=str(config_path))
            return []

        with open(config_path) as f:
            data = yaml.safe_load(f)

        examples = []
        for example_id, example_data in data.get("examples", {}).items():
            examples.append(ExampleROI(
                id=example_data.get("id", example_id),
                name=example_data.get("name", example_id),
                description=example_data.get("description", ""),
                bbox=example_data.get("bbox", {}),
                dataset=example_data.get("dataset", "mola")
            ))

        logger.info("Returned example ROIs", count=len(examples))
        return examples

    except Exception as e:
        logger.error("Failed to load example ROIs", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

