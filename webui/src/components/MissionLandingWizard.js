import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { runLandingScenario } from '../services/api';
import TerrainMap from './TerrainMap';
import SaveProjectModal from './SaveProjectModal';
import { apiFetch } from '../lib/apiBase';
export default function MissionLandingWizard() {
    const [step, setStep] = useState(1);
    const [roi, setRoi] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 });
    const [dataset, setDataset] = useState('mola');
    const [presetId, setPresetId] = useState('balanced');
    const [constraints, setConstraints] = useState({ max_slope_deg: 5.0, min_area_km2: 0.5 });
    const [presets, setPresets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [showSaveModal, setShowSaveModal] = useState(false);
    // Fetch presets on mount
    useEffect(() => {
        apiFetch('/analysis/presets?scope=site')
            .then(res => res.json())
            .then(data => {
            const sitePresets = data.site_presets || [];
            setPresets(sitePresets);
            if (sitePresets.length > 0 && !presetId) {
                setPresetId(sitePresets[0].id);
            }
        })
            .catch(err => console.error('Failed to load presets:', err));
    }, []);
    const handleRunScenario = async () => {
        setLoading(true);
        try {
            const request = {
                roi,
                dataset,
                preset_id: presetId,
                constraints,
                suitability_threshold: 0.7,
            };
            const response = await runLandingScenario(request);
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
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-center gap-4", children: [_jsxs("div", { className: `flex items-center ${step >= 1 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 1 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "1" }), _jsx("span", { className: "ml-2", children: "ROI & Dataset" })] }), _jsx("div", { className: "w-16 h-0.5 bg-gray-700" }), _jsxs("div", { className: `flex items-center ${step >= 2 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 2 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "2" }), _jsx("span", { className: "ml-2", children: "Constraints" })] }), _jsx("div", { className: "w-16 h-0.5 bg-gray-700" }), _jsxs("div", { className: `flex items-center ${step >= 3 ? 'text-blue-400' : 'text-gray-500'}`, children: [_jsx("div", { className: `w-8 h-8 rounded-full flex items-center justify-center ${step >= 3 ? 'bg-blue-600' : 'bg-gray-700'}`, children: "3" }), _jsx("span", { className: "ml-2", children: "Results" })] })] }), step === 1 && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 1: Select Region of Interest and Dataset" }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Min" }), _jsx("input", { type: "number", value: roi.lat_min, onChange: (e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Max" }), _jsx("input", { type: "number", value: roi.lat_max, onChange: (e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Min" }), _jsx("input", { type: "number", value: roi.lon_min, onChange: (e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Max" }), _jsx("input", { type: "number", value: roi.lon_max, onChange: (e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Dataset" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: [_jsx("option", { value: "mola", children: "MOLA" }), _jsx("option", { value: "hirise", children: "HiRISE" }), _jsx("option", { value: "ctx", children: "CTX" })] })] }), _jsx("button", { onClick: () => setStep(2), className: "w-full bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold", children: "Next: Mission Constraints" })] })), step === 2 && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 2: Mission Constraints and Preset Selection" }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Preset" }), _jsx("select", { value: presetId, onChange: (e) => setPresetId(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: presets.map(preset => (_jsx("option", { value: preset.id, children: preset.name }, preset.id))) }), presets.find(p => p.id === presetId) && (_jsx("p", { className: "text-sm text-gray-400 mt-1", children: presets.find(p => p.id === presetId)?.description }))] }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Max Slope (degrees)" }), _jsx("input", { type: "number", value: constraints.max_slope_deg, onChange: (e) => setConstraints({ ...constraints, max_slope_deg: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.5", min: "0", max: "90" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Min Area (km\u00B2)" }), _jsx("input", { type: "number", value: constraints.min_area_km2, onChange: (e) => setConstraints({ ...constraints, min_area_km2: parseFloat(e.target.value) }), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1", min: "0" })] })] })] }), _jsxs("div", { className: "flex gap-4", children: [_jsx("button", { onClick: () => setStep(1), className: "flex-1 bg-gray-700 hover:bg-gray-600 p-3 rounded font-semibold", children: "Back" }), _jsx("button", { onClick: handleRunScenario, disabled: loading, className: "flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold", children: loading ? 'Running Scenario...' : 'Run Scenario' })] })] })), step === 3 && results && (_jsxs("div", { className: "space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Step 3: Results" }), results.top_site && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h4", { className: "font-semibold text-lg mb-2", children: "Top Site" }), _jsxs("div", { className: "grid grid-cols-2 gap-4 text-sm", children: [_jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Site ID:" }), " ", results.top_site.site_id] }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Score:" }), " ", results.top_site.suitability_score.toFixed(3)] }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Area:" }), " ", results.top_site.area_km2.toFixed(2), " km\u00B2"] }), _jsxs("div", { children: [_jsx("span", { className: "text-gray-400", children: "Location:" }), " ", results.top_site.lat.toFixed(2), "\u00B0N, ", results.top_site.lon.toFixed(2), "\u00B0E"] })] })] })), _jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsxs("h4", { className: "font-semibold mb-2", children: ["Ranked Sites (", results.sites.length, " total)"] }), _jsx("div", { className: "max-h-64 overflow-y-auto", children: _jsxs("table", { className: "w-full text-sm", children: [_jsx("thead", { children: _jsxs("tr", { className: "border-b border-gray-600", children: [_jsx("th", { className: "text-left p-2", children: "Rank" }), _jsx("th", { className: "text-left p-2", children: "Site ID" }), _jsx("th", { className: "text-left p-2", children: "Score" }), _jsx("th", { className: "text-left p-2", children: "Area (km\u00B2)" }), _jsx("th", { className: "text-left p-2", children: "Slope (\u00B0)" })] }) }), _jsx("tbody", { children: results.sites.slice(0, 10).map(site => (_jsxs("tr", { className: "border-b border-gray-600", children: [_jsx("td", { className: "p-2", children: site.rank }), _jsx("td", { className: "p-2", children: site.site_id }), _jsx("td", { className: "p-2", children: site.suitability_score.toFixed(3) }), _jsx("td", { className: "p-2", children: site.area_km2.toFixed(2) }), _jsx("td", { className: "p-2", children: site.mean_slope_deg.toFixed(2) })] }, site.site_id))) })] }) })] }), _jsx("div", { className: "h-96", children: _jsx(TerrainMap, { roi: roi, dataset: dataset, showSites: true, showWaypoints: false }) }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: () => setShowSaveModal(true), className: "flex-1 bg-green-600 hover:bg-green-700 p-3 rounded font-semibold", children: "Save as Project" }), _jsx("button", { onClick: () => {
                                    setStep(1);
                                    setResults(null);
                                }, className: "flex-1 bg-blue-600 hover:bg-blue-700 p-3 rounded font-semibold", children: "Start New Scenario" })] })] })), showSaveModal && results && (_jsx(SaveProjectModal, { roi: roi, dataset: dataset, presetId: presetId, selectedSites: results.top_site ? [results.top_site.site_id] : [], onClose: () => setShowSaveModal(false), onSave: () => setShowSaveModal(false) }))] }));
}
