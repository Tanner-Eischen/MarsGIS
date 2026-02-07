import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { downloadDEM } from '../services/api';
import { Database, Folder, CheckCircle, Download, Cloud } from 'lucide-react';
// --- Components for Tabs (Previously Separate Pages) ---
function DataDownloadComponent() {
    const [dataset, setDataset] = useState('mola');
    const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 });
    const [force, setForce] = useState(false);
    const [errorMessage, setErrorMessage] = useState(null);
    const mutation = useMutation({
        mutationFn: (request) => downloadDEM(request),
        onSuccess: (data) => {
            setErrorMessage(null);
            alert(`Download ${data.cached ? 'completed (cached)' : 'started'}! Size: ${data.size_mb || 'N/A'} MB`);
        },
        onError: (error) => {
            const errorDetail = error.response?.data?.detail || error.message;
            setErrorMessage(errorDetail);
        },
    });
    const handleSubmit = (e) => {
        e.preventDefault();
        mutation.mutate({
            dataset,
            roi: [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max],
            force,
        });
    };
    return (_jsxs("div", { className: "glass-panel p-6 rounded-lg max-w-2xl mx-auto", children: [_jsxs("h2", { className: "text-xl font-bold text-mars-orange mb-4 flex items-center gap-2", children: [_jsx(Cloud, { className: "w-5 h-5" }), "DATA_ACQUISITION"] }), (dataset === 'hirise' || dataset === 'ctx') && (_jsxs("div", { className: "bg-amber-900/30 border border-amber-700/50 rounded p-4 mb-4 text-xs", children: [_jsx("h3", { className: "font-bold text-amber-400 mb-1", children: "MANUAL_DOWNLOAD_REQUIRED" }), _jsx("p", { className: "text-amber-200/80 mb-2", children: "High-resolution datasets (HiRISE/CTX) require specific observation IDs." }), _jsx("ul", { className: "list-disc list-inside space-y-1 text-amber-300/70", children: dataset === 'hirise' ? (_jsxs("li", { children: ["Source: ", _jsx("a", { href: "https://www.uahirise.org/hiwish/", target: "_blank", rel: "noopener noreferrer", className: "underline hover:text-white", children: "HiRISE PDS" })] })) : (_jsxs("li", { children: ["Source: ", _jsx("a", { href: "https://ode.rsl.wustl.edu/mars/", target: "_blank", rel: "noopener noreferrer", className: "underline hover:text-white", children: "WUSTL ODE" })] })) })] })), errorMessage && (_jsxs("div", { className: "bg-red-900/30 border border-red-700/50 rounded p-4 mb-4 text-xs text-red-200", children: [_jsx("h3", { className: "font-bold text-red-400 mb-1", children: "ERROR" }), _jsx("pre", { className: "whitespace-pre-wrap", children: errorMessage })] })), _jsxs("form", { onSubmit: handleSubmit, className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-xs font-bold text-gray-500 mb-1 uppercase", children: "Dataset Source" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-800/50 border border-gray-700 text-white px-3 py-2 rounded text-sm focus:border-mars-orange focus:outline-none", children: [_jsx("option", { value: "mola", children: "MOLA (Global 463m)" }), _jsx("option", { value: "hirise", children: "HiRISE (Local 1m)" }), _jsx("option", { value: "ctx", children: "CTX (Regional 18m)" })] })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "LAT MIN" }), _jsx("input", { type: "number", value: roi.lat_min, onChange: (e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "LAT MAX" }), _jsx("input", { type: "number", value: roi.lat_max, onChange: (e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "LON MIN" }), _jsx("input", { type: "number", value: roi.lon_min, onChange: (e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "LON MAX" }), _jsx("input", { type: "number", value: roi.lon_max, onChange: (e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-sm font-mono", step: "0.1" })] })] }), _jsxs("div", { className: "flex items-center pt-2", children: [_jsx("input", { type: "checkbox", id: "force", checked: force, onChange: (e) => setForce(e.target.checked), className: "accent-mars-orange mr-2" }), _jsx("label", { htmlFor: "force", className: "text-xs text-gray-400", children: "Force re-download (Ignore cache)" })] }), _jsx("button", { type: "submit", disabled: mutation.isPending, className: "w-full bg-mars-orange hover:bg-red-600 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors disabled:opacity-50 flex items-center justify-center gap-2 mt-4", children: mutation.isPending ? 'DOWNLOADING...' : _jsxs(_Fragment, { children: [_jsx(Download, { size: 16 }), " INITIATE DOWNLOAD"] }) })] })] }));
}
function ProjectsComponent() {
    return (_jsxs("div", { className: "glass-panel p-8 rounded-lg text-center text-gray-400", children: [_jsx(Folder, { className: "w-12 h-12 mx-auto mb-4 opacity-50" }), _jsx("h3", { className: "text-lg font-bold text-white mb-2", children: "PROJECT_ARCHIVE" }), _jsx("p", { className: "text-sm", children: "Project management module is currently offline." })] }));
}
function ValidationComponent() {
    return (_jsxs("div", { className: "glass-panel p-8 rounded-lg text-center text-gray-400", children: [_jsx(CheckCircle, { className: "w-12 h-12 mx-auto mb-4 opacity-50" }), _jsx("h3", { className: "text-lg font-bold text-white mb-2", children: "SYSTEM_DIAGNOSTICS" }), _jsx("p", { className: "text-sm", children: "Validation suite is running in background mode." })] }));
}
// --- Main Page Component ---
export default function DataSettings() {
    const [activeTab, setActiveTab] = useState('data');
    return (_jsxs("div", { className: "h-full flex flex-col bg-gray-900 text-white min-h-screen", children: [_jsxs("div", { className: "bg-gray-800 border-b border-gray-700 px-6 py-4", children: [_jsxs("h1", { className: "text-2xl font-bold tracking-wider mb-6 text-white flex items-center gap-3", children: [_jsx(Database, { className: "text-mars-orange" }), "SYSTEM_DATA & SETTINGS"] }), _jsxs("div", { className: "flex space-x-4 border-b border-gray-700", children: [_jsxs("button", { onClick: () => setActiveTab('data'), className: `flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'data'
                                    ? 'border-mars-orange text-mars-orange'
                                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'}`, children: [_jsx(Cloud, { className: "w-4 h-4 mr-2" }), "Data Management"] }), _jsxs("button", { onClick: () => setActiveTab('projects'), className: `flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'projects'
                                    ? 'border-blue-500 text-blue-500'
                                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'}`, children: [_jsx(Folder, { className: "w-4 h-4 mr-2" }), "Projects"] }), _jsxs("button", { onClick: () => setActiveTab('validation'), className: `flex items-center px-4 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'validation'
                                    ? 'border-green-500 text-green-500'
                                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-500'}`, children: [_jsx(CheckCircle, { className: "w-4 h-4 mr-2" }), "Validation"] })] })] }), _jsx("div", { className: "flex-1 p-6 overflow-auto bg-gray-900", children: _jsxs("div", { className: "max-w-6xl mx-auto", children: [activeTab === 'data' && _jsx(DataDownloadComponent, {}), activeTab === 'projects' && _jsx(ProjectsComponent, {}), activeTab === 'validation' && _jsx(ValidationComponent, {})] }) })] }));
}
