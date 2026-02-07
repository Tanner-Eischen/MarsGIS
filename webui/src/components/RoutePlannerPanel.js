import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { apiFetch } from '../lib/apiBase';
export default function RoutePlannerPanel({ sites, analysisDir, onPlanRoute, loading }) {
    const [startSiteId, setStartSiteId] = useState(sites[0]?.site_id || 1);
    const [endSiteId, setEndSiteId] = useState(sites[1]?.site_id || 2);
    const [presetId, setPresetId] = useState('balanced');
    const [presets, setPresets] = useState([]);
    const [showSunControls, setShowSunControls] = useState(false);
    const [sunAzimuth, setSunAzimuth] = useState(315);
    const [sunAltitude, setSunAltitude] = useState(45);
    useEffect(() => {
        apiFetch('/analysis/presets?scope=route')
            .then(res => res.json())
            .then(data => {
            const routePresets = data.route_presets || [];
            setPresets(routePresets);
            if (routePresets.length > 0) {
                setPresetId(routePresets[0].id);
            }
        })
            .catch(err => console.error('Failed to load presets:', err));
    }, []);
    const handlePlanRoute = () => {
        onPlanRoute({
            startSiteId,
            endSiteId,
            analysisDir,
            presetId,
            sunAzimuth: showSunControls ? sunAzimuth : undefined,
            sunAltitude: showSunControls ? sunAltitude : undefined,
        });
    };
    return (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-lg font-semibold", children: "Route Planning" }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Analysis Directory" }), _jsx("input", { type: "text", value: analysisDir, readOnly: true, className: "w-full bg-gray-700 text-gray-400 px-4 py-2 rounded-md" })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Start Site" }), _jsx("select", { value: startSiteId, onChange: (e) => setStartSiteId(parseInt(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: sites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ")"] }, site.site_id))) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "End Site" }), _jsx("select", { value: endSiteId, onChange: (e) => setEndSiteId(parseInt(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: sites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ")"] }, site.site_id))) })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Route Preset" }), _jsx("select", { value: presetId, onChange: (e) => setPresetId(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: presets.map(preset => (_jsx("option", { value: preset.id, children: preset.name }, preset.id))) }), presets.find(p => p.id === presetId) && (_jsx("p", { className: "text-sm text-gray-400 mt-1", children: presets.find(p => p.id === presetId)?.description }))] }), _jsxs("div", { children: [_jsxs("button", { onClick: () => setShowSunControls(!showSunControls), className: "text-sm text-blue-400 hover:text-blue-300", children: [showSunControls ? '▼' : '▶', " Sun Position Controls"] }), showSunControls && (_jsxs("div", { className: "mt-2 space-y-2", children: [_jsxs("div", { children: [_jsxs("label", { className: "block text-xs text-gray-400 mb-1", children: ["Sun Azimuth: ", sunAzimuth, "\u00B0"] }), _jsx("input", { type: "range", min: "0", max: "360", value: sunAzimuth, onChange: (e) => setSunAzimuth(parseInt(e.target.value)), className: "w-full" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-xs text-gray-400 mb-1", children: ["Sun Altitude: ", sunAltitude, "\u00B0"] }), _jsx("input", { type: "range", min: "0", max: "90", value: sunAltitude, onChange: (e) => setSunAltitude(parseInt(e.target.value)), className: "w-full" })] })] }))] }), _jsx("button", { onClick: handlePlanRoute, disabled: loading || sites.length < 2, className: "w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold", children: loading ? 'Planning Route...' : 'Plan Route' })] }));
}
