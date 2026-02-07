import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { analyzeTerrain, getExampleROIs } from '../services/api';
import EnhancedProgressBar from '../components/EnhancedProgressBar';
import AIQueryInterface from '../components/AIQueryInterface';
import { useGeoPlan } from '../context/GeoPlanContext';
// Generate UUID for task tracking
function generateTaskId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}
export default function TerrainAnalysis({ roi: propRoi, onRoiChange, dataset: propDataset, onDatasetChange }) {
    const { publishSites } = useGeoPlan();
    const [localRoi, setLocalRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 });
    const [localDataset, setLocalDataset] = useState('mola');
    const roi = propRoi || localRoi;
    const setRoi = onRoiChange || setLocalRoi;
    const dataset = propDataset || localDataset;
    const setDataset = onDatasetChange || setLocalDataset;
    const [threshold, setThreshold] = useState(0.7);
    const [examples, setExamples] = useState([]);
    const [criteriaWeights, setCriteriaWeights] = useState(null);
    const [autoRunAfterAI, setAutoRunAfterAI] = useState(true);
    const [siteType, setSiteType] = useState('construction');
    useEffect(() => {
        const s = localStorage.getItem('terrain.roi');
        const d = localStorage.getItem('terrain.dataset');
        const t = localStorage.getItem('terrain.threshold');
        if (s && !propRoi) {
            try {
                const o = JSON.parse(s);
                if (o && o.lat_min !== undefined)
                    setRoi(o);
            }
            catch { }
        }
        if (d && !propDataset)
            setDataset(d);
        if (t)
            setThreshold(parseFloat(t));
    }, []);
    // Only save to local storage if we are controlling the state locally or if it changes
    useEffect(() => { localStorage.setItem('terrain.roi', JSON.stringify(roi)); }, [roi]);
    useEffect(() => { localStorage.setItem('terrain.dataset', dataset); }, [dataset]);
    useEffect(() => { localStorage.setItem('terrain.threshold', String(threshold)); }, [threshold]);
    const [taskId, setTaskId] = useState(null);
    const mutation = useMutation({
        mutationFn: async (request) => {
            const taskId = generateTaskId();
            const response = await analyzeTerrain({ ...request, task_id: taskId });
            setTaskId(taskId);
            return response;
        },
        onSuccess: (data) => {
            if (data.task_id) {
                setTaskId(data.task_id);
            }
            if (data.sites && data.sites.length > 0) {
                publishSites(siteType, data.sites);
            }
        },
        onError: (error) => {
            setTaskId(null);
            alert(`Analysis failed: ${error.response?.data?.detail || error.message}`);
        },
    });
    const handleAIQueryProcessed = (parameters) => {
        console.log("AI Query processed, raw parameters:", parameters);
        const newRoi = parameters.roi ? {
            lat_min: parameters.roi.min_lat,
            lat_max: parameters.roi.max_lat,
            lon_min: parameters.roi.min_lon,
            lon_max: parameters.roi.max_lon
        } : null;
        if (newRoi) {
            setRoi(newRoi);
        }
        if (parameters.dataset_preferences?.primary) {
            setDataset(parameters.dataset_preferences.primary.toLowerCase());
        }
        if (parameters.criteria_weights) {
            setCriteriaWeights(parameters.criteria_weights);
        }
        if (autoRunAfterAI) {
            setTaskId(null);
            const analysisROI = newRoi ? [newRoi.lat_min, newRoi.lat_max, newRoi.lon_min, newRoi.lon_max] : [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max];
            mutation.mutate({
                roi: analysisROI,
                dataset: parameters.dataset_preferences?.primary?.toLowerCase() ?? dataset,
                threshold,
                criteria_weights: parameters.criteria_weights ?? criteriaWeights ?? undefined,
            });
        }
    };
    const handleSubmit = (e) => {
        e.preventDefault();
        setTaskId(null);
        mutation.mutate({
            roi: [roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max],
            dataset,
            threshold,
            criteria_weights: criteriaWeights ?? undefined,
        });
    };
    const generateWeightSummary = (weights) => {
        if (!weights || Object.keys(weights).length === 0) {
            return 'No specific criteria weights are being applied. The analysis will use default, balanced priorities.';
        }
        const safetyCriteria = ['slope', 'roughness'];
        let safetyWeight = 0;
        let scienceWeight = 0;
        let topCriterion = '';
        let maxWeight = 0;
        for (const [key, value] of Object.entries(weights)) {
            if (value > maxWeight) {
                maxWeight = value;
                topCriterion = key;
            }
            if (safetyCriteria.includes(key))
                safetyWeight += value;
            else
                scienceWeight += value;
        }
        const top = topCriterion.replace(/_/g, ' ');
        if (safetyWeight > scienceWeight * 1.5)
            return `This analysis strongly prioritizes safety, with a focus on finding areas with low ${top}.`;
        if (scienceWeight > safetyWeight * 1.5)
            return `This analysis strongly prioritizes scientific value, focusing on areas with high ${top}.`;
        return `This analysis is taking a balanced approach, with the most influential factor being ${top}.`;
    };
    return (_jsxs("div", { className: "space-y-4 text-sm", children: [_jsxs("div", { className: "glass-panel p-4 rounded-lg border border-cyan-500/20", children: [_jsxs("div", { className: "mb-3 text-xs text-cyan-400/90 uppercase tracking-wide font-bold flex justify-between", children: [_jsx("span", { children: "AI Mission Assistant" }), _jsx("span", { className: "text-[10px] opacity-50", children: "NLP_MODULE_V1" })] }), _jsx(AIQueryInterface, { onQueryProcessed: handleAIQueryProcessed, className: "mb-4" }), _jsxs("div", { className: "flex items-center justify-between border-t border-gray-700/50 pt-3", children: [_jsx("div", { className: "text-[10px] text-gray-400", children: criteriaWeights ? 'AI weights loaded' : 'Standard weights' }), _jsxs("label", { className: "flex items-center space-x-2 text-[10px] text-gray-300 cursor-pointer hover:text-white", children: [_jsx("input", { type: "checkbox", checked: autoRunAfterAI, onChange: (e) => setAutoRunAfterAI(e.target.checked), className: "accent-cyan-500" }), _jsx("span", { children: "Auto-execute" })] })] })] }), taskId && (_jsx("div", { className: "mb-4", children: _jsx(EnhancedProgressBar, { taskId: taskId, title: "ANALYSIS_PROGRESS", showDetails: true, onComplete: () => setTaskId(null), onError: (error) => alert(`Progress tracking error: ${error}`) }) })), _jsxs("form", { onSubmit: handleSubmit, className: "glass-panel p-4 rounded-lg space-y-3 border border-gray-700/50", children: [_jsxs("div", { className: "grid grid-cols-2 gap-3", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-[10px] font-bold text-gray-500 mb-1 uppercase", children: "Operation Type" }), _jsxs("select", { value: siteType, onChange: (e) => setSiteType(e.target.value), className: "w-full bg-gray-800 border border-gray-600 text-white px-2 py-1.5 rounded text-xs focus:border-cyan-500 focus:outline-none", children: [_jsx("option", { value: "construction", children: "Construction" }), _jsx("option", { value: "landing", children: "Landing" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-[10px] font-bold text-gray-500 mb-1 uppercase", children: "Source Data" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-800 border border-gray-600 text-white px-2 py-1.5 rounded text-xs focus:border-cyan-500 focus:outline-none", children: [_jsx("option", { value: "mola", children: "MOLA (Global)" }), _jsx("option", { value: "hirise", children: "HiRISE (High Res)" }), _jsx("option", { value: "ctx", children: "CTX (Medium Res)" })] })] })] }), _jsxs("div", { className: "flex items-center justify-between border-b border-gray-700/50 pb-2", children: [_jsx("div", { className: "text-[10px] font-bold text-gray-400 uppercase", children: "Target Coordinates" }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("button", { type: "button", onClick: async () => { const data = await getExampleROIs(); setExamples(data); }, className: "px-2 py-1 bg-gray-800 text-[10px] text-cyan-400 border border-cyan-500/30 rounded hover:bg-gray-700", children: "Load ROI" }), examples.length > 0 && (_jsxs("select", { onChange: (e) => {
                                            const sel = examples.find(x => x.id === e.target.value);
                                            if (sel) {
                                                setRoi(sel.bbox);
                                                setDataset(sel.dataset);
                                            }
                                        }, className: "bg-gray-800 text-white px-2 py-1 rounded text-[10px] border border-gray-600 max-w-[100px]", defaultValue: "", children: [_jsx("option", { value: "", disabled: true, children: "Select..." }), examples.map(x => (_jsx("option", { value: x.id, children: x.name }, x.id)))] }))] })] }), _jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-[10px] text-gray-500 mb-0.5", children: "LAT MIN" }), _jsx("input", { type: "number", value: roi.lat_min, onChange: (e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-[10px] text-gray-500 mb-0.5", children: "LAT MAX" }), _jsx("input", { type: "number", value: roi.lat_max, onChange: (e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-[10px] text-gray-500 mb-0.5", children: "LON MIN" }), _jsx("input", { type: "number", value: roi.lon_min, onChange: (e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-[10px] text-gray-500 mb-0.5", children: "LON MAX" }), _jsx("input", { type: "number", value: roi.lon_max, onChange: (e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) }), className: "w-full bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs font-mono focus:border-cyan-500 focus:outline-none", step: "0.1" })] })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-[10px] font-bold text-gray-400 mb-1 uppercase", children: ["Suitability Threshold: ", threshold] }), _jsx("input", { type: "range", min: "0", max: "1", step: "0.1", value: threshold, onChange: (e) => setThreshold(parseFloat(e.target.value)), className: "w-full accent-cyan-500 h-1.5" })] }), criteriaWeights && (_jsxs("div", { className: "space-y-2 p-2 bg-gray-900/50 rounded border border-gray-700/50", children: [_jsx("h4", { className: "text-[10px] font-bold text-gray-400 uppercase", children: "Active Weights" }), _jsx("p", { className: "text-[10px] text-gray-500 italic truncate", children: generateWeightSummary(criteriaWeights || {}) }), _jsx("div", { className: "grid grid-cols-2 gap-1 text-[10px]", children: Object.entries(criteriaWeights).map(([key, value]) => (_jsxs("div", { className: "flex justify-between items-center bg-gray-800 px-2 py-0.5 rounded", children: [_jsx("span", { className: "font-mono text-gray-400 truncate max-w-[70px]", children: key.replace(/_/g, ' ').toUpperCase() }), _jsx("span", { className: "font-bold text-cyan-300", children: Number(value).toFixed(2) })] }, key))) })] })), _jsx("button", { type: "submit", disabled: mutation.isPending, className: "w-full bg-cyan-700 hover:bg-cyan-600 text-white px-4 py-2 rounded text-xs font-bold tracking-wider transition-colors disabled:opacity-50 shadow-[0_0_15px_rgba(6,182,212,0.2)]", children: mutation.isPending ? 'PROCESSING...' : 'INITIATE ANALYSIS' })] }), mutation.data && (_jsxs("div", { className: "glass-panel p-4 rounded-lg border border-green-500/20", children: [_jsx("h3", { className: "text-xs font-bold mb-2 text-green-400 tracking-wider uppercase", children: "Results Matrix" }), _jsxs("div", { className: "mb-3 flex gap-2", children: [_jsxs("div", { className: "bg-gray-900 p-2 rounded border border-gray-700 flex-1", children: [_jsx("span", { className: "block text-[10px] text-gray-500 uppercase", children: "Sites Found" }), _jsx("span", { className: "text-lg font-bold text-white", children: mutation.data.sites.length })] }), _jsxs("div", { className: "bg-gray-900 p-2 rounded border border-gray-700 flex-1", children: [_jsx("span", { className: "block text-[10px] text-gray-500 uppercase", children: "Top Score" }), _jsx("span", { className: "text-lg font-bold text-green-400", children: mutation.data.top_site_score.toFixed(3) })] })] }), _jsx("div", { className: "overflow-x-auto max-h-40 custom-scrollbar", children: _jsxs("table", { className: "w-full text-[10px] font-mono", children: [_jsx("thead", { className: "sticky top-0 bg-gray-900", children: _jsxs("tr", { className: "border-b border-gray-700 text-gray-500", children: [_jsx("th", { className: "text-left p-1", children: "ID" }), _jsx("th", { className: "text-left p-1", children: "RANK" }), _jsx("th", { className: "text-left p-1", children: "AREA" }), _jsx("th", { className: "text-left p-1", children: "SCORE" })] }) }), _jsx("tbody", { children: mutation.data.sites.map((site) => (_jsxs("tr", { className: "border-b border-gray-800 hover:bg-gray-800/50", children: [_jsx("td", { className: "p-1 text-cyan-300", children: site.site_id }), _jsx("td", { className: "p-1 text-white", children: site.rank }), _jsx("td", { className: "p-1 text-gray-400", children: site.area_km2.toFixed(1) }), _jsx("td", { className: "p-1 text-green-400 font-bold", children: site.suitability_score.toFixed(2) })] }, site.site_id))) })] }) })] }))] }));
}
