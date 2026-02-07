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
            const taskId = generateTaskId();
            const response = await planNavigation({ ...request, task_id: taskId });
            setTaskId(taskId);
            return response;
        },
        onSuccess: (data) => {
            if (data.task_id) {
                setTaskId(data.task_id);
            }
        },
        onError: (error) => {
            setTaskId(null);
            alert(`Navigation planning failed: ${error.response?.data?.detail || error.message}`);
        },
    });
    const handleSubmit = (e) => {
        e.preventDefault();
        setTaskId(null);
        mutation.mutate({
            site_id: siteId,
            start_lat: startLat,
            start_lon: startLon,
            strategy,
            analysis_dir: analysisDir,
        });
    };
    return (_jsxs("div", { className: "space-y-6 text-sm", children: [_jsxs("div", { className: "glass-panel p-6 rounded-lg space-y-3", children: [_jsx("div", { className: "text-xs font-bold text-gray-400 uppercase mb-2", children: "Start Point Selection" }), _jsxs("div", { className: "bg-gray-900/50 p-3 rounded border border-gray-700", children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "AVAILABLE SITES" }), _jsxs("select", { value: landingSites.find(s => Math.abs(s.lat - startLat) < 1e-6 && Math.abs(s.lon - startLon) < 1e-6)?.site_id || '', onChange: (e) => {
                                    const id = parseInt(e.target.value);
                                    const site = landingSites.find(s => s.site_id === id);
                                    if (site) {
                                        setStartLat(site.lat);
                                        setStartLon(site.lon);
                                    }
                                }, className: "w-full bg-gray-800 text-white px-3 py-2 rounded text-sm border border-gray-600 focus:border-cyan-500 focus:outline-none", children: [_jsx("option", { value: "", disabled: true, children: landingSites.length > 0 ? 'Select starting site...' : 'No landing sites found' }), landingSites.map(site => (_jsxs("option", { value: site.site_id, children: ["Site ", site.site_id, " (Rank ", site.rank, ", Score ", site.suitability_score.toFixed(2), ")"] }, site.site_id)))] })] })] }), (landingSites.length > 0 || constructionSites.length > 0) && (_jsxs("div", { className: "glass-panel p-6 rounded-lg", children: [_jsx("h3", { className: "text-sm font-bold text-cyan-400 uppercase mb-4", children: "AI Recommendations" }), _jsx(MLSiteRecommendation, { candidateSites: (landingSites.length > 0 ? landingSites : constructionSites).map(s => ({
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
                        } }), mlRecs && mlRecs.length > 0 && (_jsx("div", { className: "mt-4 flex items-center gap-2", children: _jsx("button", { type: "button", onClick: () => {
                                const top = mlRecs[0];
                                if (!top)
                                    return;
                                setStartLat(top.coordinates[0]);
                                setStartLon(top.coordinates[1]);
                                const pool = landingSites.length > 0 ? landingSites : constructionSites;
                                const match = pool.find(s => Math.abs(s.lat - top.coordinates[0]) < 1e-6 && Math.abs(s.lon - top.coordinates[1]) < 1e-6);
                                setRecommendedLandingSiteId(match ? match.site_id : null);
                            }, className: "bg-cyan-700 hover:bg-cyan-600 text-white px-3 py-2 rounded text-xs font-bold uppercase", children: "USE TOP RECOMMENDATION" }) }))] })), _jsxs("form", { onSubmit: handleSubmit, className: "glass-panel p-6 rounded-lg space-y-4", children: [_jsxs("div", { className: "flex items-center justify-between border-b border-gray-700/50 pb-2", children: [_jsx("div", { className: "text-xs font-bold text-gray-400 uppercase", children: "Example Seed" }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("button", { type: "button", onClick: async () => { const data = await getExampleROIs(); setExamples(data); }, className: "px-2 py-1 bg-gray-800 text-xs text-cyan-400 border border-cyan-500/30 rounded hover:bg-gray-700", children: "Load Examples" }), examples.length > 0 && (_jsxs("select", { onChange: (e) => {
                                            const sel = examples.find(x => x.id === e.target.value);
                                            if (sel) {
                                                const cLat = (sel.bbox.lat_min + sel.bbox.lat_max) / 2;
                                                const cLon = (sel.bbox.lon_min + sel.bbox.lon_max) / 2;
                                                setStartLat(cLat);
                                                setStartLon(cLon);
                                            }
                                        }, className: "bg-gray-800 text-white px-2 py-1 rounded text-xs border border-gray-600", defaultValue: "", children: [_jsx("option", { value: "", disabled: true, children: "Select..." }), examples.map(x => (_jsx("option", { value: x.id, children: x.name }, x.id)))] }))] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "TARGET SITE ID" }), _jsx("input", { type: "number", value: siteId, onChange: (e) => setSiteId(parseInt(e.target.value)), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none", min: "1" })] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "START LAT" }), _jsx("input", { type: "number", value: startLat, onChange: (e) => setStartLat(parseFloat(e.target.value)), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "START LON" }), _jsx("input", { type: "number", value: startLon, onChange: (e) => setStartLon(parseFloat(e.target.value)), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-3 py-2 rounded text-sm font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-500 mb-1", children: "STRATEGY" }), _jsxs("select", { value: strategy, onChange: (e) => setStrategy(e.target.value), className: "w-full bg-gray-800 border border-gray-600 text-white px-3 py-2 rounded text-sm focus:border-cyan-500 focus:outline-none", children: [_jsx("option", { value: "safest", children: "Safest (Risk Averse)" }), _jsx("option", { value: "balanced", children: "Balanced (Standard)" }), _jsx("option", { value: "direct", children: "Direct (Distance Priority)" })] })] }), _jsx("button", { type: "submit", disabled: mutation.isPending, className: "w-full bg-green-600 hover:bg-green-500 text-white px-4 py-3 rounded font-bold tracking-wider transition-colors disabled:opacity-50 shadow-[0_0_15px_rgba(34,197,94,0.2)]", children: mutation.isPending ? 'CALCULATING...' : 'GENERATE FLIGHTPATH' })] }), mutation.data && (_jsxs("div", { className: "glass-panel p-6 rounded-lg", children: [_jsx("h3", { className: "text-lg font-bold mb-4 text-green-400 tracking-wider", children: "FLIGHTPATH_DATA" }), mutation.data.path_length_m && (_jsxs("div", { className: "bg-gray-900 p-3 rounded border border-gray-700 mb-4 flex justify-between items-center", children: [_jsx("span", { className: "text-xs text-gray-500", children: "EST. DISTANCE" }), _jsxs("span", { className: "text-lg font-mono text-white", children: [(mutation.data.path_length_m).toFixed(2), " m"] })] })), _jsx("div", { className: "overflow-x-auto max-h-60 custom-scrollbar", children: _jsxs("table", { className: "w-full text-xs font-mono", children: [_jsx("thead", { className: "sticky top-0 bg-gray-900", children: _jsxs("tr", { className: "border-b border-gray-700 text-gray-500", children: [_jsx("th", { className: "text-left p-2", children: "WP" }), _jsx("th", { className: "text-left p-2", children: "X" }), _jsx("th", { className: "text-left p-2", children: "Y" }), _jsx("th", { className: "text-left p-2", children: "TOL" })] }) }), _jsx("tbody", { children: mutation.data.waypoints.map((waypoint) => (_jsxs("tr", { className: "border-b border-gray-800 hover:bg-gray-800/50", children: [_jsx("td", { className: "p-2 text-green-300", children: waypoint.waypoint_id }), _jsx("td", { className: "p-2 text-gray-400", children: waypoint.x_meters.toFixed(1) }), _jsx("td", { className: "p-2 text-gray-400", children: waypoint.y_meters.toFixed(1) }), _jsx("td", { className: "p-2 text-gray-500", children: waypoint.tolerance_meters.toFixed(1) })] }, waypoint.waypoint_id))) })] }) })] }))] }));
}
