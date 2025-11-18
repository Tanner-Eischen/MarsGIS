"""Project management for saving and loading analyses."""

from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from marshab.types import BoundingBox
from marshab.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Project:
    """Project data model."""
    id: str
    name: str
    description: str
    created_at: str
    roi: Dict[str, float]
    dataset: str
    preset_id: Optional[str]
    selected_sites: List[int]
    routes: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Project":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProjectSummary:
    """Project summary for listing."""
    id: str
    name: str
    created_at: str
    description: str


class ProjectManager:
    """Manages project storage and retrieval."""
    
    def __init__(self, projects_dir: Optional[Path] = None):
        """Initialize project manager.
        
        Args:
            projects_dir: Directory to store projects (default: data/projects)
        """
        if projects_dir is None:
            from marshab.config import get_config
            config = get_config()
            projects_dir = config.paths.data_dir / "projects"
        
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized ProjectManager", projects_dir=str(self.projects_dir))
    
    def save_project(self, project: Project) -> str:
        """Save project to disk.
        
        Args:
            project: Project to save
            
        Returns:
            Project ID
        """
        project_file = self.projects_dir / f"{project.id}.json"
        
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project.to_dict(), f, indent=2)
        
        logger.info("Saved project", project_id=project.id, name=project.name)
        return project.id
    
    def load_project(self, project_id: str) -> Project:
        """Load project from disk.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Loaded Project
            
        Raises:
            FileNotFoundError: If project not found
        """
        project_file = self.projects_dir / f"{project_id}.json"
        
        if not project_file.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        
        with open(project_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        project = Project.from_dict(data)
        logger.info("Loaded project", project_id=project_id, name=project.name)
        return project
    
    def list_projects(self) -> List[ProjectSummary]:
        """List all projects.
        
        Returns:
            List of project summaries
        """
        projects = []
        
        for project_file in self.projects_dir.glob("*.json"):
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                projects.append(ProjectSummary(
                    id=data["id"],
                    name=data["name"],
                    created_at=data["created_at"],
                    description=data.get("description", "")
                ))
            except Exception as e:
                logger.warning(f"Failed to load project {project_file}", error=str(e))
        
        # Sort by created_at (newest first)
        projects.sort(key=lambda p: p.created_at, reverse=True)
        
        logger.info("Listed projects", count=len(projects))
        return projects
    
    def delete_project(self, project_id: str) -> None:
        """Delete project.
        
        Args:
            project_id: Project identifier
            
        Raises:
            FileNotFoundError: If project not found
        """
        project_file = self.projects_dir / f"{project_id}.json"
        
        if not project_file.exists():
            raise FileNotFoundError(f"Project not found: {project_id}")
        
        project_file.unlink()
        logger.info("Deleted project", project_id=project_id)

