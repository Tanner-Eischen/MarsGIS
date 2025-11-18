"""Unit tests for project management."""

import pytest
from pathlib import Path
from datetime import datetime

from marshab.core.projects import ProjectManager, Project, ProjectSummary
from marshab.types import BoundingBox


class TestProjectManager:
    """Tests for ProjectManager."""
    
    def test_save_and_load_project(self, tmp_path):
        """Test saving and loading a project."""
        manager = ProjectManager(projects_dir=tmp_path / "projects")
        
        project = Project(
            id="test_project",
            name="Test Project",
            description="Test description",
            created_at=datetime.now().isoformat(),
            roi={"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
            dataset="mola",
            preset_id="balanced",
            selected_sites=[1, 2, 3],
            routes=[],
            metadata={}
        )
        
        project_id = manager.save_project(project)
        assert project_id == "test_project"
        
        loaded = manager.load_project("test_project")
        assert loaded.id == project.id
        assert loaded.name == project.name
        assert loaded.selected_sites == [1, 2, 3]
    
    def test_list_projects(self, tmp_path):
        """Test listing projects."""
        manager = ProjectManager(projects_dir=tmp_path / "projects")
        
        # Create multiple projects
        for i in range(3):
            project = Project(
                id=f"project_{i}",
                name=f"Project {i}",
                description=f"Description {i}",
                created_at=datetime.now().isoformat(),
                roi={"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
                dataset="mola",
                preset_id=None,
                selected_sites=[],
                routes=[],
                metadata={}
            )
            manager.save_project(project)
        
        summaries = manager.list_projects()
        assert len(summaries) == 3
        assert all(isinstance(s, ProjectSummary) for s in summaries)
    
    def test_delete_project(self, tmp_path):
        """Test deleting a project."""
        manager = ProjectManager(projects_dir=tmp_path / "projects")
        
        project = Project(
            id="to_delete",
            name="To Delete",
            description="",
            created_at=datetime.now().isoformat(),
            roi={"lat_min": 40.0, "lat_max": 41.0, "lon_min": 180.0, "lon_max": 181.0},
            dataset="mola",
            preset_id=None,
            selected_sites=[],
            routes=[],
            metadata={}
        )
        
        manager.save_project(project)
        assert len(manager.list_projects()) == 1
        
        manager.delete_project("to_delete")
        assert len(manager.list_projects()) == 0
        
        with pytest.raises(FileNotFoundError):
            manager.delete_project("to_delete")

