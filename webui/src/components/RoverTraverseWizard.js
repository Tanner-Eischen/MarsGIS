import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { useGeoPlan } from '../context/GeoPlanContext';
import { planMultipleRoutes } from '../services/api';
import TerrainMap from './TerrainMap';
import { apiFetch } from '../lib/apiBase';
export default function RoverTraverseWizard() {
    const [step, setStep] = useState(1);
    const [startSiteId, setStartSiteId] = useState(1);
    const [endSiteId, setEndSiteId] = useState(2);
    const [analysisDir, setAnalysisDir] = useState('data/output');
    const [presetId, setPresetId] = useState('balanced');
    const [roverCapabilities, setRoverCapabilities] = useState({ max_slope_deg: 25.0, max_roughness: 1.0 });
    const [presets, setPresets] = useState([]);
    const [sites, setSites] = useState([]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const { recommendedLandingSiteId } = useGeoPlan();
    // Fetch route presets on mount
    useEffect(() => {
        apiFetch('/analysis/presets?scope=route')
            .then(res => res.json())
            .then(data => {
            const routePresets = data.route_presets || [];
            setPresets(routePresets);
            if (routePresets.length > 0 && !presetId) {
                setPresetId(routePresets[0].id);
            }
        })
            .catch(err => console.error('Failed to load presets:', err));
    }, []);
    // Load sites from analysis results
    useEffect(() => {
        if (analysisDir) {
            apiFetch('/visualization/sites-geojson')
                .then(res => res.json())
                .then(data => {
                const siteList = data.features.map((f) => ({
                    site_id: f.properties.site_id,
                    lat: f.geometry.type === 'Point' ? f.geometry.coordinates[1] : f.properties.lat || 0,
                    lon: f.geometry.type === 'Point' ? f.geometry.coordinates[0] : f.properties.lon || 0,
                    suitability_score: f.properties.suitability_score,
                    rank: f.properties.rank,
                }));
                setSites(siteList);
                if (siteList.length > 0) {
                    if (!startSiteId)
                        setStartSiteId(siteList[0].site_id);
                    if (!endSiteId && siteList.length > 1)
                        setEndSiteId(siteList[1].site_id);
                }
            })
                .catch(err => console.error('Failed to load sites:', err));
        }
    }, [analysisDir]);
    useEffect(() => {
        if (recommendedLandingSiteId)
            setStartSiteId(recommendedLandingSiteId);
    }, [recommendedLandingSiteId]);
    const handleRunScenario = async () => {
        setLoading(true);
        try {
            const startSite = sites.find(s => s.site_id === startSiteId);
            const endSite = sites.find(s => s.site_id === endSiteId);
            if (!startSite || !endSite)
                throw new Error('Invalid start or end site');
            const request = {
                site_id: endSite.site_id,
                analysis_dir: analysisDir,
                start_lat: startSite.lat,
                start_lon: startSite.lon,
                strategies: ['safest', 'balanced', 'direct'],
                max_slope_deg: roverCapabilities.max_slope_deg,
            };
            const response = await planMultipleRoutes(request);
            setResults(response);
            setStep(3);
        }
        catch (error) {
            console.error('Scenario failed:', error);
            alert(`Scenario failed: ${error.response?.data?.detail || error.message}`);
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-center gap-4", children: [_jsxs("div", { className: `flex items-center ${step >= 1 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "1" }), _jsx("span", { className: "ml-2", children: "Select Sites" })] }), _jsx("div", { className: "w-16 h-0.5 bg-gray-700" }), _jsxs("div", { className: `flex items-center ${step >= 2 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "2" }), _jsx("span", { className: "ml-2", children: "Rover & Route" })] }), _jsx("div", { className: "w-16 h-0.5 bg-gray-700" }), _jsxs("div", { className: `flex items-center ${step >= 3 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "3" }), _jsx("span", { className: "ml-2", children: "Route & Cost" })] })] }), step === 1 && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 1: Select Start and End Sites" }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Analysis Directory" }), _jsx("input", { type: "text", value: analysisDir, onChange: (e) => setAnalysisDir(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", placeholder: "data/output" })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Start Site" }), _jsx("select", { value: startSiteId, onChange: (e) => setStartSiteId(parseInt(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: sites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ", Score ", site.suitability_score.toFixed(2), ")"] }, site.site_id))) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "End Site" }), _jsx("select", { value: endSiteId, onChange: (e) => setEndSiteId(parseInt(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: sites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ", Score ", site.suitability_score.toFixed(2), ")"] }, site.site_id))) })] })] }), _jsx("button", { onClick: () => setStep(2), disabled: sites.length === 0, className: "w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold", children: "Next: Rover Capabilities" })] })), step === 2 && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 2: Rover Capabilities and Route Preset" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Route Preset" }), _jsx("select", { value: presetId, onChange: (e) => setPresetId(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: presets.map(preset => (_jsx("option", { value: preset.id, children: preset.name }, preset.id))) }), presets.find(p => p.id === presetId) && (_jsx("p", { className: "text-sm text-gray-400 mt-1", children: presets.find(p => p.id === presetId)?.description }))] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Max Slope (degrees)" }), _jsx("input", { type: "number", value: roverCapabilities.max_slope_deg, onChange: (e) => setRoverCapabilities({ ...roverCapabilities, max_slope_deg: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "1", min: "0", max: "90" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Max Roughness" }), _jsx("input", { type: "number", value: roverCapabilities.max_roughness, onChange: (e) => setRoverCapabilities({ ...roverCapabilities, max_roughness: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1", min: "0" })] })] })] }), _jsxs("div", { className: "flex gap-4", children: [_jsx("button", { onClick: () => setStep(1), className: "flex-1 bg-gray-700 hover:bg-gray-600 p-3 rounded font-semibold", children: "Back" }), _jsx("button", { onClick: handleRunScenario, disabled: loading, className: "flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold", children: loading ? 'Planning Route...' : 'Plan Route' })] })] })), step === 3 && results && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 3: Route and Cost Summary" }), _jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h4", { className: "font-semibold text-lg mb-2", children: "Route Metrics" }), _jsx("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4 text-sm", children: results.routes.map((r) => (_jsxs("div", { className: "p-3 rounded border", style: { borderColor: r.strategy === 'safest' ? '#00ff00' : r.strategy === 'balanced' ? '#1e90ff' : '#ffa500' }, children: [_jsx("div", { className: "font-semibold capitalize", children: r.strategy }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Distance:" }), " ", r.total_distance_m.toFixed(2), " m"] }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Waypoints:" }), " ", r.num_waypoints] }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Relative Cost:" }), " ", r.relative_cost_percent.toFixed(1), "%"] })] }, r.strategy))) })] }), _jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h4", { className: "font-semibold mb-2", children: "Waypoints (showing up to 20 per route)" }), _jsx("div", { className: "max-h-64 overflow-y-auto", children: _jsxs("table", { className: "w-full text-sm", children: [_jsx("thead", { children: _jsxs("tr", { className: "border-b border-gray-600", children: [_jsx("th", { className: "text-left p-2", children: "Route" }), _jsx("th", { className: "text-left p-2", children: "ID" }), _jsx("th", { className: "text-left p-2", children: "X (m)" }), _jsx("th", { className: "text-left p-2", children: "Y (m)" }), _jsx("th", { className: "text-left p-2", children: "Tolerance (m)" })] }) }), _jsx("tbody", { children: results.routes.flatMap(r => r.waypoints.slice(0, 20).map(wp => (_jsxs("tr", { className: "border-b border-gray-600", children: [_jsx("td", { className: "p-2 capitalize", children: r.strategy }), _jsx("td", { className: "p-2", children: wp.waypoint_id }), _jsx("td", { className: "p-2", children: wp.x_meters.toFixed(2) }), _jsx("td", { className: "p-2", children: wp.y_meters.toFixed(2) }), _jsx("td", { className: "p-2", children: wp.tolerance_meters.toFixed(2) })] }, `${r.strategy}-${wp.waypoint_id}`)))) })] }) })] }), _jsx("div", { className: "h-96", children: _jsx(TerrainMap, { roi: { lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 }, dataset: "mola", showSites: true, showWaypoints: true }) }), _jsx("button", { onClick: () => {
                            setStep(1);
                            setResults(null);
                        }, className: "w-full bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold", children: "Plan New Route" })] }))] }));
}
