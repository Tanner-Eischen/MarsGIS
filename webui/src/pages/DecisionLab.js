import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import PresetsSelector from '../components/PresetsSelector';
import AdvancedWeightsPanel from '../components/AdvancedWeightsPanel';
import SiteScoresList from '../components/SiteScoresList';
import ExplainabilityPanel from '../components/ExplainabilityPanel';
import TerrainMap from '../components/TerrainMap';
import SaveProjectModal from '../components/SaveProjectModal';
import ExamplesDrawer from '../components/ExamplesDrawer';
import { apiFetch } from '../lib/apiBase';
export default function DecisionLab() {
    const [roi, setROI] = useState({ lat_min: 40.0, lat_max: 41.0, lon_min: 180.0, lon_max: 181.0 });
    const [dataset, setDataset] = useState('mola');
    const [selectedPreset, setSelectedPreset] = useState('balanced');
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [customWeights, setCustomWeights] = useState({});
    const [sites, setSites] = useState([]);
    const [selectedSite, setSelectedSite] = useState(null);
    const [loading, setLoading] = useState(false);
    const [showSaveModal, setShowSaveModal] = useState(false);
    const [showExamples, setShowExamples] = useState(false);
    const [explainMap, setExplainMap] = useState(false);
    // Fetch presets on mount
    const [presets, setPresets] = useState([]);
    useEffect(() => {
        apiFetch('/analysis/presets')
            .then(res => res.json())
            .then(data => setPresets(data.site_presets || []))
            .catch(err => console.error('Failed to load presets:', err));
    }, []);
    // Run analysis when ROI or preset changes
    const runAnalysis = async () => {
        if (!roi)
            return;
        setLoading(true);
        try {
            const request = {
                roi,
                dataset,
                preset_id: selectedPreset,
                custom_weights: Object.keys(customWeights).length > 0 ? customWeights : null,
                threshold: 0.6
            };
            const response = await apiFetch('/analysis/site-scores', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });
            if (response.ok) {
                const data = await response.json();
                setSites(data);
            }
            else {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                console.error('Analysis failed:', errorData.detail || response.statusText);
                alert(`Analysis failed: ${errorData.detail || response.statusText}`);
            }
        }
        catch (error) {
            console.error('Analysis error:', error);
            alert(`Analysis error: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsxs("div", { className: "flex flex-col bg-gray-900 text-gray-100", style: { height: 'calc(100vh - 8rem)' }, children: [_jsx("div", { className: "bg-gray-800 border-b border-gray-700 p-4 mb-4", children: _jsxs("div", { className: "flex justify-between items-center", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-2xl font-bold", children: "Mars Landing Site Decision Lab" }), _jsx("p", { className: "text-gray-400 text-sm", children: "Explore landing sites using preset criteria or customize your own" })] }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: () => setShowExamples(true), className: "bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded font-semibold text-sm", children: "Load Example" }), _jsxs("label", { className: "flex items-center gap-2 bg-gray-700 px-4 py-2 rounded cursor-pointer", children: [_jsx("input", { type: "checkbox", checked: explainMap, onChange: (e) => setExplainMap(e.target.checked), className: "w-4 h-4" }), _jsx("span", { className: "text-sm", children: "Explain this map" })] })] })] }) }), _jsxs("div", { className: "flex-1 flex overflow-hidden", children: [_jsxs("div", { className: "w-96 bg-gray-800 border-r border-gray-700 overflow-y-auto", children: [_jsxs("div", { className: "p-4 border-b border-gray-700", children: [_jsx("h3", { className: "font-semibold mb-2", children: "Region of Interest" }), _jsxs("div", { className: "space-y-2 text-sm", children: [_jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsx("input", { type: "number", placeholder: "Lat Min", value: roi?.lat_min || '', className: "bg-gray-700 p-2 rounded text-white", onChange: (e) => setROI(prev => ({ ...prev, lat_min: parseFloat(e.target.value) })) }), _jsx("input", { type: "number", placeholder: "Lat Max", value: roi?.lat_max || '', className: "bg-gray-700 p-2 rounded text-white", onChange: (e) => setROI(prev => ({ ...prev, lat_max: parseFloat(e.target.value) })) }), _jsx("input", { type: "number", placeholder: "Lon Min", value: roi?.lon_min || '', className: "bg-gray-700 p-2 rounded text-white", onChange: (e) => setROI(prev => ({ ...prev, lon_min: parseFloat(e.target.value) })) }), _jsx("input", { type: "number", placeholder: "Lon Max", value: roi?.lon_max || '', className: "bg-gray-700 p-2 rounded text-white", onChange: (e) => setROI(prev => ({ ...prev, lon_max: parseFloat(e.target.value) })) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-xs text-gray-400 mb-1", children: "Dataset" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-700 p-2 rounded text-white text-sm", children: [_jsx("option", { value: "mola", children: "MOLA" }), _jsx("option", { value: "hirise", children: "HiRISE" }), _jsx("option", { value: "ctx", children: "CTX" })] })] })] })] }), _jsx(PresetsSelector, { presets: presets, selected: selectedPreset, onSelect: setSelectedPreset }), _jsxs("div", { className: "p-4 border-b border-gray-700", children: [_jsxs("button", { onClick: () => setShowAdvanced(!showAdvanced), className: "text-sm text-blue-400 hover:text-blue-300 flex items-center gap-2", children: [_jsx("span", { children: showAdvanced ? '▼' : '▶' }), _jsx("span", { children: "Advanced Weights" })] }), showAdvanced && (_jsx(AdvancedWeightsPanel, { weights: customWeights, onChange: setCustomWeights, presetWeights: presets.find(p => p.id === selectedPreset)?.weights || {} }))] }), _jsxs("div", { className: "p-4 space-y-2", children: [_jsx("button", { onClick: runAnalysis, disabled: !roi || loading, className: "w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed p-3 rounded font-semibold", children: loading ? 'Analyzing...' : 'Run Analysis' }), sites.length > 0 && (_jsxs(_Fragment, { children: [_jsx("button", { onClick: () => setShowSaveModal(true), className: "w-full bg-green-600 hover:bg-green-700 p-3 rounded font-semibold", children: "Save as Project" }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: async () => {
                                                            try {
                                                                const response = await apiFetch('/export/suitability-geotiff', {
                                                                    method: 'POST',
                                                                    headers: { 'Content-Type': 'application/json' },
                                                                    body: JSON.stringify({
                                                                        roi,
                                                                        dataset,
                                                                        preset_id: selectedPreset
                                                                    })
                                                                });
                                                                if (response.ok) {
                                                                    const blob = await response.blob();
                                                                    const url = window.URL.createObjectURL(blob);
                                                                    const a = document.createElement('a');
                                                                    a.href = url;
                                                                    a.download = `suitability_${Date.now()}.tif`;
                                                                    a.click();
                                                                    window.URL.revokeObjectURL(url);
                                                                }
                                                                else {
                                                                    alert('Export failed');
                                                                }
                                                            }
                                                            catch (error) {
                                                                console.error('Export error:', error);
                                                                alert('Export failed');
                                                            }
                                                        }, className: "flex-1 bg-purple-600 hover:bg-purple-700 p-2 rounded text-sm font-semibold", children: "Export GeoTIFF" }), _jsx("button", { onClick: async () => {
                                                            try {
                                                                const response = await apiFetch('/export/report', {
                                                                    method: 'POST',
                                                                    headers: { 'Content-Type': 'application/json' },
                                                                    body: JSON.stringify({ format: 'markdown' })
                                                                });
                                                                if (response.ok) {
                                                                    const blob = await response.blob();
                                                                    const url = window.URL.createObjectURL(blob);
                                                                    const a = document.createElement('a');
                                                                    a.href = url;
                                                                    a.download = `analysis_report_${Date.now()}.md`;
                                                                    a.click();
                                                                    window.URL.revokeObjectURL(url);
                                                                }
                                                                else {
                                                                    alert('Report generation failed');
                                                                }
                                                            }
                                                            catch (error) {
                                                                console.error('Report error:', error);
                                                                alert('Report generation failed');
                                                            }
                                                        }, className: "flex-1 bg-orange-600 hover:bg-orange-700 p-2 rounded text-sm font-semibold", children: "Export Report" })] })] }))] }), sites.length > 0 && (_jsx(SiteScoresList, { sites: sites, selectedSite: selectedSite, onSelectSite: setSelectedSite }))] }), _jsxs("div", { className: "flex-1 flex flex-col", children: [_jsx("div", { className: "flex-1 p-4", children: roi && (_jsx(TerrainMap, { roi: roi, dataset: dataset, showSites: sites.length > 0, showWaypoints: false, onSiteSelect: setSelectedSite, selectedSiteId: selectedSite })) }), selectedSite !== null && (_jsx("div", { className: "p-4 border-t border-gray-700", children: _jsx(ExplainabilityPanel, { site: sites.find(s => s.site_id === selectedSite) || null, weights: presets.find(p => p.id === selectedPreset)?.weights || {} }) }))] })] }), showSaveModal && (_jsx(SaveProjectModal, { roi: roi, dataset: dataset, presetId: selectedPreset, selectedSites: selectedSite ? [selectedSite] : sites.slice(0, 5).map(s => s.site_id), onClose: () => setShowSaveModal(false), onSave: () => setShowSaveModal(false) })), showExamples && (_jsx(ExamplesDrawer, { isOpen: showExamples, onClose: () => setShowExamples(false), onSelectExample: (example) => {
                    setROI(example.bbox);
                    setDataset(example.dataset);
                } })), explainMap && (_jsxs("div", { className: "fixed bottom-4 right-4 bg-gray-800 border border-gray-700 rounded-lg p-4 max-w-md z-40", children: [_jsx("h4", { className: "font-semibold mb-2", children: "Map Explanation" }), _jsx("p", { className: "text-sm text-gray-300 mb-2", children: "Colors represent suitability scores (0-1), where higher values indicate better landing sites." }), _jsxs("p", { className: "text-sm text-gray-300 mb-2", children: ["Current preset: ", _jsx("span", { className: "font-semibold", children: presets.find(p => p.id === selectedPreset)?.name || selectedPreset })] }), _jsxs("p", { className: "text-sm text-gray-300", children: ["This preset optimizes for: ", presets.find(p => p.id === selectedPreset)?.description || 'balanced criteria'] })] }))] }));
}
