import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
export default function SaveProjectModal({ roi, dataset, presetId, selectedSites, onClose, onSave }) {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const handleSave = async () => {
        if (!name.trim()) {
            alert('Project name is required');
            return;
        }
        const projectData = {
            name,
            description,
            roi,
            dataset,
            preset_id: presetId,
            selected_sites: selectedSites,
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
                onClose();
            }
            else {
                const error = await response.json();
                alert(`Failed to save project: ${error.detail || 'Unknown error'}`);
            }
        }
        catch (error) {
            console.error('Failed to save project:', error);
            alert('Failed to save project');
        }
    };
    return (_jsx("div", { className: "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50", children: _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 w-full max-w-md", children: [_jsx("h3", { className: "text-xl font-semibold mb-4", children: "Save as Project" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Project Name" }), _jsx("input", { type: "text", value: name, onChange: (e) => setName(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", placeholder: "My Mars Mission" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Description" }), _jsx("textarea", { value: description, onChange: (e) => setDescription(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", rows: 3, placeholder: "Project description..." })] }), _jsx("div", { className: "text-xs text-gray-400", children: "This will save the current ROI, dataset, preset, and selected sites." })] }), _jsxs("div", { className: "flex gap-2 mt-6", children: [_jsx("button", { onClick: onClose, className: "flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded font-semibold", children: "Cancel" }), _jsx("button", { onClick: handleSave, className: "flex-1 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold", children: "Save Project" })] })] }) }));
}
