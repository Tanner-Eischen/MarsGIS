import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useQuery, useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getStatus, getDemoMode, setDemoMode } from '../services/api';
export default function Dashboard() {
    const { data: status, isLoading, error, isError } = useQuery({
        queryKey: ['status'],
        queryFn: getStatus,
        retry: 1,
        retryDelay: 1000,
    });
    const { data: demoMode } = useQuery({
        queryKey: ['demo-mode'],
        queryFn: getDemoMode,
    });
    const toggleMutation = useMutation({
        mutationFn: async (enabled) => setDemoMode(enabled),
    });
    const [toggleMessage, setToggleMessage] = useState(null);
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h2", { className: "text-2xl font-bold mb-4", children: "Welcome to MarsHab" }), _jsx("p", { className: "text-gray-300 mb-4", children: "Mars Habitat Site Selection and Rover Navigation System" }), _jsxs("div", { className: "flex space-x-4", children: [_jsx(Link, { to: "/download", className: "bg-mars-orange hover:bg-mars-red text-white px-4 py-2 rounded-md transition-colors", children: "Download DEM Data" }), _jsx(Link, { to: "/analyze", className: "bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md transition-colors", children: "Analyze Terrain" }), _jsx(Link, { to: "/navigate", className: "bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors", children: "Plan Navigation" })] }), _jsxs("div", { className: "mt-4 flex items-center space-x-3", children: [_jsx("span", { className: "text-sm text-gray-300", children: "Demo Mode" }), _jsx("button", { onClick: async () => {
                                    const next = !demoMode?.enabled;
                                    try {
                                        await toggleMutation.mutateAsync(next);
                                        setToggleMessage(next ? 'Demo mode enabled' : 'Demo mode disabled');
                                        setTimeout(() => setToggleMessage(null), 2000);
                                    }
                                    catch { }
                                }, className: `px-3 py-1 rounded-md text-sm ${demoMode?.enabled ? 'bg-green-600' : 'bg-gray-600'} text-white`, children: demoMode?.enabled ? 'On' : 'Off' }), toggleMessage && (_jsx("span", { className: "text-xs text-gray-400 ml-2", children: toggleMessage }))] })] }), isLoading ? (_jsxs("div", { className: "text-center py-8", children: [_jsx("div", { className: "text-lg mb-2", children: "Loading system status..." }), _jsx("div", { className: "text-sm text-gray-400", children: "Connecting to backend at http://localhost:5000" })] })) : isError ? (_jsxs("div", { className: "bg-red-900 border border-red-700 rounded-lg p-6", children: [_jsx("h3", { className: "text-lg font-semibold text-red-200 mb-2", children: "\u26A0\uFE0F Backend Connection Error" }), _jsx("p", { className: "text-red-300 mb-2", children: "Could not connect to the backend server at http://localhost:5000" }), _jsxs("p", { className: "text-sm text-red-400 mb-4", children: ["Error: ", error instanceof Error ? error.message : 'Unknown error'] }), _jsxs("div", { className: "text-sm text-gray-400", children: [_jsx("p", { children: "Make sure the backend server is running:" }), _jsx("code", { className: "block bg-gray-800 p-2 rounded mt-2", children: "poetry run python -m marshab.web.server" })] })] })) : status ? (_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-6", children: [_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-lg font-semibold mb-2", children: "Cache Status" }), _jsxs("p", { className: "text-gray-400", children: ["Files: ", status.cache.file_count] }), _jsxs("p", { className: "text-gray-400", children: ["Size: ", status.cache.size_mb, " MB"] })] }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-lg font-semibold mb-2", children: "Output Files" }), _jsxs("p", { className: "text-gray-400", children: ["Count: ", status.output.file_count] })] }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-lg font-semibold mb-2", children: "Data Sources" }), _jsx("p", { className: "text-gray-400", children: status.config.data_sources.join(', ') })] })] })) : null] }));
}
