"""Project management endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.core.projects import Project, ProjectManager
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

# Global project manager instance
project_manager = ProjectManager()


class ProjectCreateRequest(BaseModel):
    """Request to create a project."""
    name: str = Field(..., description="Project name")
    description: str = Field("", description="Project description")
    roi: dict[str, float] = Field(..., description="ROI bounding box")
    dataset: str = Field(..., description="Dataset name")
    preset_id: Optional[str] = Field(None, description="Preset ID used")
    selected_sites: list[int] = Field(default_factory=list, description="Selected site IDs")
    routes: list[dict] = Field(default_factory=list, description="Saved routes")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ProjectResponse(BaseModel):
    """Project response."""
    id: str
    name: str
    description: str
    created_at: str
    roi: dict[str, float]
    dataset: str
    preset_id: Optional[str]
    selected_sites: list[int]
    routes: list[dict]
    metadata: dict


class ProjectSummaryResponse(BaseModel):
    """Project summary response."""
    id: str
    name: str
    created_at: str
    description: str


@router.get("", response_model=list[ProjectSummaryResponse])
async def list_projects():
    """List all projects."""
    try:
        summaries = project_manager.list_projects()
        return [
            ProjectSummaryResponse(
                id=s.id,
                name=s.name,
                created_at=s.created_at,
                description=s.description
            )
            for s in summaries
        ]
    except Exception as e:
        logger.error("Failed to list projects", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=ProjectResponse)
async def create_project(request: ProjectCreateRequest):
    """Create a new project."""
    try:
        # Generate project ID
        project_id = f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        project = Project(
            id=project_id,
            name=request.name,
            description=request.description,
            created_at=datetime.now().isoformat(),
            roi=request.roi,
            dataset=request.dataset,
            preset_id=request.preset_id,
            selected_sites=request.selected_sites,
            routes=request.routes,
            metadata=request.metadata
        )

        project_manager.save_project(project)

        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            roi=project.roi,
            dataset=project.dataset,
            preset_id=project.preset_id,
            selected_sites=project.selected_sites,
            routes=project.routes,
            metadata=project.metadata
        )
    except Exception as e:
        logger.error("Failed to create project", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get project by ID."""
    try:
        project = project_manager.load_project(project_id)
        return ProjectResponse(
            id=project.id,
            name=project.name,
            description=project.description,
            created_at=project.created_at,
            roi=project.roi,
            dataset=project.dataset,
            preset_id=project.preset_id,
            selected_sites=project.selected_sites,
            routes=project.routes,
            metadata=project.metadata
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    except Exception as e:
        logger.error("Failed to get project", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{project_id}")
async def delete_project(project_id: str):
    """Delete project."""
    try:
        project_manager.delete_project(project_id)
        return {"status": "deleted", "project_id": project_id}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    except Exception as e:
        logger.error("Failed to delete project", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

