import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { downloadDEM } from '../services/api';
export default function DataDownload() {
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
    return (_jsxs("div", { className: "space-y-6", children: [_jsx("h2", { className: "text-3xl font-bold", children: "Download DEM Data" }), (dataset === 'hirise' || dataset === 'ctx') && (_jsxs("div", { className: "bg-yellow-900 border border-yellow-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-yellow-300 mb-2", children: "Manual Download Required" }), _jsx("p", { className: "text-yellow-200 text-sm mb-2", children: dataset === 'hirise'
                            ? 'HiRISE (1m resolution) datasets require manual download. They are region-specific and require selecting specific observation IDs.'
                            : 'CTX (18m resolution) datasets require manual download. They are region-specific and require selecting specific observation IDs.' }), _jsx("p", { className: "text-yellow-200 text-sm mb-2", children: "Sources:" }), _jsx("ul", { className: "text-yellow-200 text-sm list-disc list-inside space-y-1", children: dataset === 'hirise' ? (_jsxs(_Fragment, { children: [_jsx("li", { children: _jsx("a", { href: "https://www.uahirise.org/hiwish/", target: "_blank", rel: "noopener noreferrer", className: "underline", children: "HiRISE PDS" }) }), _jsx("li", { children: _jsx("a", { href: "https://s3.amazonaws.com/mars-hirise-pds/", target: "_blank", rel: "noopener noreferrer", className: "underline", children: "AWS S3 HiRISE Archive" }) })] })) : (_jsx("li", { children: _jsx("a", { href: "https://ode.rsl.wustl.edu/mars/", target: "_blank", rel: "noopener noreferrer", className: "underline", children: "WUSTL ODE Mars Data" }) })) }), _jsx("p", { className: "text-yellow-200 text-sm mt-2", children: "After downloading, place the DEM file in the cache directory or the system will use it automatically if it matches the expected pattern." })] })), errorMessage && (_jsxs("div", { className: "bg-red-900 border border-red-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-red-300 mb-2", children: "Download Error" }), _jsx("pre", { className: "text-red-200 text-sm whitespace-pre-wrap", children: errorMessage })] })), _jsxs("form", { onSubmit: handleSubmit, className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Dataset" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: [_jsx("option", { value: "mola", children: "MOLA (463m resolution)" }), _jsx("option", { value: "hirise", children: "HiRISE (1m resolution)" }), _jsx("option", { value: "ctx", children: "CTX (18m resolution)" })] })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Min" }), _jsx("input", { type: "number", value: roi.lat_min, onChange: (e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Max" }), _jsx("input", { type: "number", value: roi.lat_max, onChange: (e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Min" }), _jsx("input", { type: "number", value: roi.lon_min, onChange: (e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Max" }), _jsx("input", { type: "number", value: roi.lon_max, onChange: (e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] })] }), _jsxs("div", { className: "flex items-center", children: [_jsx("input", { type: "checkbox", id: "force", checked: force, onChange: (e) => setForce(e.target.checked), className: "mr-2" }), _jsx("label", { htmlFor: "force", className: "text-sm", children: "Force re-download" })] }), _jsx("button", { type: "submit", disabled: mutation.isPending, className: "w-full bg-mars-orange hover:bg-mars-red text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50", children: mutation.isPending ? 'Downloading...' : 'Download DEM' })] })] }));
}
