import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
export default function Projects() {
    const [projects, setProjects] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const navigate = useNavigate();
    useEffect(() => {
        loadProjects();
    }, []);
    const loadProjects = async () => {
        try {
            const response = await fetch('http://localhost:5000/api/v1/projects');
            if (response.ok) {
                const data = await response.json();
                setProjects(data);
            }
        }
        catch (error) {
            console.error('Failed to load projects:', error);
        }
        finally {
            setLoading(false);
        }
    };
    const handleOpenProject = async (projectId) => {
        try {
            const response = await fetch(`http://localhost:5000/api/v1/projects/${projectId}`);
            if (response.ok) {
                const project = await response.json();
                // Navigate to Decision Lab with project data
                navigate('/decision-lab', { state: { project } });
            }
        }
        catch (error) {
            console.error('Failed to open project:', error);
            alert('Failed to open project');
        }
    };
    const handleDeleteProject = async (projectId) => {
        if (!confirm('Are you sure you want to delete this project?')) {
            return;
        }
        try {
            const response = await fetch(`http://localhost:5000/api/v1/projects/${projectId}`, {
                method: 'DELETE'
            });
            if (response.ok) {
                loadProjects();
            }
            else {
                alert('Failed to delete project');
            }
        }
        catch (error) {
            console.error('Failed to delete project:', error);
            alert('Failed to delete project');
        }
    };
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex justify-between items-center", children: [_jsx("h2", { className: "text-3xl font-bold", children: "Projects" }), _jsx("button", { onClick: () => setShowCreateModal(true), className: "bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold", children: "New Project" })] }), loading ? (_jsx("div", { className: "text-center py-8", children: "Loading projects..." })) : projects.length === 0 ? (_jsx("div", { className: "text-center py-8 text-gray-400", children: "No projects yet. Create one to save your analyses." })) : (_jsx("div", { className: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4", children: projects.map(project => (_jsxs("div", { className: "bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-blue-500 transition-colors", children: [_jsx("h3", { className: "text-xl font-semibold mb-2", children: project.name }), _jsx("p", { className: "text-sm text-gray-400 mb-4 line-clamp-2", children: project.description || 'No description' }), _jsxs("div", { className: "text-xs text-gray-500 mb-4", children: ["Created: ", new Date(project.created_at).toLocaleDateString()] }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: () => handleOpenProject(project.id), className: "flex-1 bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm font-semibold", children: "Open" }), _jsx("button", { onClick: () => handleDeleteProject(project.id), className: "bg-red-600 hover:bg-red-700 px-3 py-2 rounded text-sm font-semibold", children: "Delete" })] })] }, project.id))) })), showCreateModal && (_jsx(CreateProjectModal, { onClose: () => setShowCreateModal(false), onSave: () => {
                    setShowCreateModal(false);
                    loadProjects();
                } }))] }));
}
function CreateProjectModal({ onClose, onSave }) {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const handleSave = async () => {
        if (!name.trim()) {
            alert('Project name is required');
            return;
        }
        // Get current analysis state (simplified - would get from context/state)
        const projectData = {
            name,
            description,
            roi: { lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 },
            dataset: 'mola',
            preset_id: 'balanced',
            selected_sites: [],
            routes: [],
            metadata: {}
        };
        try {
            const response = await fetch('http://localhost:5000/api/v1/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(projectData)
            });
            if (response.ok) {
                onSave();
            }
            else {
                alert('Failed to create project');
            }
        }
        catch (error) {
            console.error('Failed to create project:', error);
            alert('Failed to create project');
        }
    };
    return (_jsx("div", { className: "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50", children: _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 w-full max-w-md", children: [_jsx("h3", { className: "text-xl font-semibold mb-4", children: "Create New Project" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Project Name" }), _jsx("input", { type: "text", value: name, onChange: (e) => setName(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", placeholder: "My Mars Mission" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Description" }), _jsx("textarea", { value: description, onChange: (e) => setDescription(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", rows: 3, placeholder: "Project description..." })] })] }), _jsxs("div", { className: "flex gap-2 mt-6", children: [_jsx("button", { onClick: onClose, className: "flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded font-semibold", children: "Cancel" }), _jsx("button", { onClick: handleSave, className: "flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold", children: "Save" })] })] }) }));
}
