import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
export default function Validation() {
    const [healthStatus, setHealthStatus] = useState(null);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        checkHealth();
        const interval = setInterval(checkHealth, 5000); // Check every 5 seconds
        return () => clearInterval(interval);
    }, []);
    const checkHealth = async () => {
        try {
            const [, readyResponse] = await Promise.all([
                fetch('http://localhost:5000/api/v1/health/live'),
                fetch('http://localhost:5000/api/v1/health/ready')
            ]);
            if (readyResponse.ok) {
                const data = await readyResponse.json();
                setHealthStatus(data);
            }
        }
        catch (error) {
            console.error('Health check failed:', error);
            setHealthStatus({
                status: 'error',
                checks: { error: 'Failed to connect to server' }
            });
        }
        finally {
            setLoading(false);
        }
    };
    const getStatusColor = (status) => {
        switch (status) {
            case 'ready':
            case 'alive':
                return 'text-green-400';
            case 'not_ready':
                return 'text-red-400';
            default:
                return 'text-yellow-400';
        }
    };
    return (_jsxs("div", { className: "space-y-6", children: [_jsx("h2", { className: "text-3xl font-bold", children: "System Validation" }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-xl font-semibold mb-4", children: "Health Status" }), loading ? (_jsx("div", { className: "text-center py-4", children: "Checking health..." })) : healthStatus ? (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "flex items-center gap-2", children: [_jsx("span", { className: "text-sm font-medium", children: "Overall Status:" }), _jsx("span", { className: `font-semibold ${getStatusColor(healthStatus.status)}`, children: healthStatus.status.toUpperCase() })] }), _jsxs("div", { className: "border-t border-gray-700 pt-4", children: [_jsx("h4", { className: "text-sm font-semibold mb-2", children: "Component Checks:" }), _jsx("div", { className: "space-y-2", children: Object.entries(healthStatus.checks).map(([component, status]) => (_jsxs("div", { className: "flex justify-between items-center", children: [_jsxs("span", { className: "text-sm text-gray-400 capitalize", children: [component.replace('_', ' '), ":"] }), _jsx("span", { className: `text-sm font-medium ${getStatusColor(status)}`, children: status })] }, component))) })] })] })) : (_jsx("div", { className: "text-center py-4 text-red-400", children: "Failed to check health" }))] })] }));
}
