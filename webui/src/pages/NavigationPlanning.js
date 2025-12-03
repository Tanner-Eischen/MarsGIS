import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { planNavigation, getExampleROIs } from '../services/api';
import MLSiteRecommendation from '../components/MLSiteRecommendation';
import { useGeoPlan } from '../context/GeoPlanContext';
// Generate UUID for task tracking
function generateTaskId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
export default function NavigationPlanning() {
    const { landingSites, constructionSites, setRecommendedLandingSiteId } = useGeoPlan();
    const [siteId, setSiteId] = useState(1);
    const [startLat, setStartLat] = useState(40.0);
    const [startLon, setStartLon] = useState(180.0);
    const [strategy, setStrategy] = useState('balanced');
    const [analysisDir, setAnalysisDir] = useState('data/output');
    const [taskId, setTaskId] = useState(null);
    const [examples, setExamples] = useState([]);
    const [mlRecs, setMlRecs] = useState(null);
    useEffect(() => {
        const sLat = localStorage.getItem('nav.startLat');
        const sLon = localStorage.getItem('nav.startLon');
        const dir = localStorage.getItem('nav.analysisDir');
        if (sLat)
            setStartLat(parseFloat(sLat));
        if (sLon)
            setStartLon(parseFloat(sLon));
        if (dir)
            setAnalysisDir(dir);
    }, []);
    useEffect(() => { localStorage.setItem('nav.startLat', String(startLat)); }, [startLat]);
    useEffect(() => { localStorage.setItem('nav.startLon', String(startLon)); }, [startLon]);
    useEffect(() => { localStorage.setItem('nav.analysisDir', analysisDir); }, [analysisDir]);
    const mutation = useMutation({
        mutationFn: async (request) => {
            // Generate task_id before making request
            const taskId = generateTaskId();
            // Add task_id to request
            const response = await planNavigation({ ...request, task_id: taskId });
            // Only set taskId after successful API call so WebSocket can connect
            setTaskId(taskId);
            return response;
        },
        onSuccess: (data) => {
            // Use task_id from response if available, otherwise keep the one we generated
            if (data.task_id) {
                setTaskId(data.task_id);
            }
        },
        onError: (error) => {
            // Clear taskId on error to stop WebSocket connection attempts
            setTaskId(null);
            alert(`Navigation planning failed: ${error.response?.data?.detail || error.message}`);
        },
    });
    const handleSubmit = (e) => {
        e.preventDefault();
        setTaskId(null); // Reset task_id for new request
        mutation.mutate({
            site_id: siteId,
            start_lat: startLat,
            start_lon: startLon,
            strategy,
            analysis_dir: analysisDir,
        });
    };
    return (_jsxs("div", { className: "space-y-6", children: [_jsx("h2", { className: "text-3xl font-bold", children: "Navigation Planning" }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 space-y-3", children: [_jsx("div", { className: "text-sm text-gray-300", children: "Select a starting site from the list, or use mission-type ranking to explore options." }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Starting Site (Landing)" }), _jsxs("select", { value: landingSites.find(s => Math.abs(s.lat - startLat) < 1e-6 && Math.abs(s.lon - startLon) < 1e-6)?.site_id || '', onChange: (e) => {
                                    const id = parseInt(e.target.value);
                                    const site = landingSites.find(s => s.site_id === id);
                                    if (site) {
                                        setStartLat(site.lat);
                                        setStartLon(site.lon);
                                    }
                                }, className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: [_jsx("option", { value: "", disabled: true, children: landingSites.length > 0 ? 'Select starting site' : 'No landing sites found — run Terrain Analysis' }), landingSites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ", Score ", site.suitability_score.toFixed(2), ")"] }, site.site_id)))] })] })] }), (landingSites.length > 0 || constructionSites.length > 0) && (_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-xl font-semibold mb-4", children: "ML Site Recommendation" }), _jsx(MLSiteRecommendation, { candidateSites: (landingSites.length > 0 ? landingSites : constructionSites).map(s => ({
                            coordinates: [s.lat, s.lon],
                            elevation: s.mean_elevation_m,
                            slope_mean: s.mean_slope_deg,
                            aspect_mean: 180.0,
                            roughness_mean: s.mean_roughness,
                            tri_mean: 20.0
                        })), destinationCoordinates: (constructionSites.find(s => s.site_id === siteId)) ? [
                            constructionSites.find(s => s.site_id === siteId).lat,
                            constructionSites.find(s => s.site_id === siteId).lon
                        ] : undefined, onRecommendationComplete: (recs) => {
                            setMlRecs(recs || []);
                        } }), mlRecs && mlRecs.length > 0 && (_jsxs("div", { className: "mt-4 flex items-center gap-2", children: [_jsx("button", { type: "button", onClick: () => {
                                    const top = mlRecs[0];
                                    if (!top)
                                        return;
                                    setStartLat(top.coordinates[0]);
                                    setStartLon(top.coordinates[1]);
                                    const pool = landingSites.length > 0 ? landingSites : constructionSites;
                                    const match = pool.find(s => Math.abs(s.lat - top.coordinates[0]) < 1e-6 && Math.abs(s.lon - top.coordinates[1]) < 1e-6);
                                    setRecommendedLandingSiteId(match ? match.site_id : null);
                                }, className: "bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded", children: "Apply top recommendation as starting site" }), _jsx("span", { className: "text-xs text-gray-400", children: "Or select a starting site from the list above." })] })), _jsx("div", { className: "mt-3 text-xs text-gray-400", children: "Use mission type to explore options. Select a site above to apply as the starting location." })] })), _jsxs("form", { onSubmit: handleSubmit, className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("div", { className: "text-sm text-gray-300", children: "Seed from Example ROI" }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("button", { type: "button", onClick: async () => { const data = await getExampleROIs(); setExamples(data); }, className: "px-3 py-2 bg-gray-700 text-white rounded", children: "Load Examples" }), examples.length > 0 && (_jsxs("select", { onChange: (e) => {
                                            const sel = examples.find(x => x.id === e.target.value);
                                            if (sel) {
                                                const cLat = (sel.bbox.lat_min + sel.bbox.lat_max) / 2;
                                                const cLon = (sel.bbox.lon_min + sel.bbox.lon_max) / 2;
                                                setStartLat(cLat);
                                                setStartLon(cLon);
                                            }
                                        }, className: "bg-gray-700 text-white px-2 py-2 rounded", defaultValue: "", children: [_jsx("option", { value: "", disabled: true, children: "Select example" }), examples.map(x => (_jsx("option", { value: x.id, children: x.name }, x.id)))] }))] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Target Site ID" }), _jsx("input", { type: "number", value: siteId, onChange: (e) => setSiteId(parseInt(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", min: "1" })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Start Latitude" }), _jsx("input", { type: "number", value: startLat, onChange: (e) => setStartLat(parseFloat(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Start Longitude" }), _jsx("input", { type: "number", value: startLon, onChange: (e) => setStartLon(parseFloat(e.target.value)), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Pathfinding Strategy" }), _jsxs("select", { value: strategy, onChange: (e) => setStrategy(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: [_jsx("option", { value: "safest", children: "Safest (prioritize safety)" }), _jsx("option", { value: "balanced", children: "Balanced (default)" }), _jsx("option", { value: "direct", children: "Direct (prioritize distance)" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Analysis Directory" }), _jsx("input", { type: "text", value: analysisDir, onChange: (e) => setAnalysisDir(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md" })] }), _jsx("button", { type: "submit", disabled: mutation.isPending, className: "w-full bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors disabled:opacity-50", children: mutation.isPending ? 'Planning...' : 'Generate Waypoints' })] }), mutation.data && (_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsxs("h3", { className: "text-xl font-semibold mb-4", children: ["Navigation Plan (", mutation.data.num_waypoints, " waypoints)"] }), mutation.data.path_length_m && (_jsxs("p", { className: "text-gray-300 mb-4", children: ["Estimated Path Length: ", mutation.data.path_length_m.toFixed(2), " meters"] })), _jsx("div", { className: "overflow-x-auto", children: _jsxs("table", { className: "w-full text-sm", children: [_jsx("thead", { children: _jsxs("tr", { className: "border-b border-gray-700", children: [_jsx("th", { className: "text-left p-2", children: "Waypoint ID" }), _jsx("th", { className: "text-left p-2", children: "X (meters)" }), _jsx("th", { className: "text-left p-2", children: "Y (meters)" }), _jsx("th", { className: "text-left p-2", children: "Tolerance (meters)" })] }) }), _jsx("tbody", { children: mutation.data.waypoints.map((waypoint) => (_jsxs("tr", { className: "border-b border-gray-700", children: [_jsx("td", { className: "p-2", children: waypoint.waypoint_id }), _jsx("td", { className: "p-2", children: waypoint.x_meters.toFixed(2) }), _jsx("td", { className: "p-2", children: waypoint.y_meters.toFixed(2) }), _jsx("td", { className: "p-2", children: waypoint.tolerance_meters.toFixed(2) })] }, waypoint.waypoint_id))) })] }) })] }))] }));
}
