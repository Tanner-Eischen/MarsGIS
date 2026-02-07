import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { analyzeSolarPotential, getExampleROIs } from '../services/api';
export default function SolarAnalysis({ roi: propRoi, onRoiChange, dataset: propDataset, onDatasetChange }) {
    const [localRoi, setLocalRoi] = useState({
        lat_min: 18.0,
        lat_max: 18.6,
        lon_min: 77.0,
        lon_max: 77.8,
    });
    const [localDataset, setLocalDataset] = useState('mola');
    const roi = propRoi || localRoi;
    const setRoi = onRoiChange || setLocalRoi;
    const dataset = propDataset || localDataset;
    const setDataset = onDatasetChange || setLocalDataset;
    const [examples, setExamples] = useState([]);
    useEffect(() => {
        const s = localStorage.getItem('solar.roi');
        const d = localStorage.getItem('solar.dataset');
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
    }, []);
    useEffect(() => { localStorage.setItem('solar.roi', JSON.stringify(roi)); }, [roi]);
    useEffect(() => { localStorage.setItem('solar.dataset', dataset); }, [dataset]);
    // Panel configuration
    const [panelEfficiency, setPanelEfficiency] = useState(0.25);
    const [panelArea, setPanelArea] = useState(100.0);
    // Mission parameters
    const [batteryCapacity, setBatteryCapacity] = useState(50.0);
    const [dailyPowerNeeds, setDailyPowerNeeds] = useState(20.0);
    const [batteryCostPerKwh] = useState(1000.0);
    const [missionDuration, setMissionDuration] = useState(500.0);
    // Sun position (static for now)
    const [sunAzimuth, setSunAzimuth] = useState(0.0);
    const [sunAltitude, setSunAltitude] = useState(45.0);
    // Results
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const handleAnalyze = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await analyzeSolarPotential({
                roi,
                dataset,
                sun_azimuth: sunAzimuth,
                sun_altitude: sunAltitude,
                panel_efficiency: panelEfficiency,
                panel_area_m2: panelArea,
                battery_capacity_kwh: batteryCapacity,
                daily_power_needs_kwh: dailyPowerNeeds,
                battery_cost_per_kwh: batteryCostPerKwh,
                mission_duration_days: missionDuration,
            });
            setResults(response);
        }
        catch (err) {
            console.error('Solar analysis failed:', err);
            setError(err instanceof Error ? err.message : 'Failed to analyze solar potential');
        }
        finally {
            setLoading(false);
        }
    };
    return (_jsxs("div", { className: "space-y-6 text-sm", children: [_jsxs("div", { className: "glass-panel p-4 rounded-lg border border-amber-500/20", children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsx("h3", { className: "text-xs font-bold text-amber-400 uppercase tracking-wide", children: "Analysis Configuration" }), _jsx("button", { onClick: handleAnalyze, disabled: loading, className: "px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white rounded text-xs font-bold uppercase tracking-wider disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_10px_rgba(245,158,11,0.2)]", children: loading ? 'CALCULATING...' : 'RUN ANALYSIS' })] }), error && (_jsx("div", { className: "p-2 mb-3 bg-red-900/50 border border-red-700 rounded text-xs text-red-300", children: error })), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("label", { className: "text-[10px] font-bold text-gray-500 uppercase", children: "Region of Interest" }), _jsx("button", { onClick: async () => { const data = await getExampleROIs(); setExamples(data); }, className: "text-[10px] text-amber-400 hover:text-amber-300 underline", children: "Load Example" })] }), examples.length > 0 && (_jsxs("select", { onChange: (e) => {
                                            const sel = examples.find(x => x.id === e.target.value);
                                            if (sel) {
                                                setRoi(sel.bbox);
                                                setDataset(sel.dataset);
                                            }
                                        }, className: "w-full bg-gray-800 text-white px-2 py-1 rounded text-[10px] border border-gray-600 mb-2", defaultValue: "", children: [_jsx("option", { value: "", disabled: true, children: "Select..." }), examples.map(x => (_jsx("option", { value: x.id, children: x.name }, x.id)))] })), _jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsx("input", { type: "number", value: roi.lat_min, onChange: (e) => setRoi({ ...roi, lat_min: parseFloat(e.target.value) }), className: "bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs", step: "0.1", placeholder: "Lat Min" }), _jsx("input", { type: "number", value: roi.lat_max, onChange: (e) => setRoi({ ...roi, lat_max: parseFloat(e.target.value) }), className: "bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs", step: "0.1", placeholder: "Lat Max" }), _jsx("input", { type: "number", value: roi.lon_min, onChange: (e) => setRoi({ ...roi, lon_min: parseFloat(e.target.value) }), className: "bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs", step: "0.1", placeholder: "Lon Min" }), _jsx("input", { type: "number", value: roi.lon_max, onChange: (e) => setRoi({ ...roi, lon_max: parseFloat(e.target.value) }), className: "bg-gray-900/50 border border-gray-700 text-white px-2 py-1 rounded text-xs", step: "0.1", placeholder: "Lon Max" })] })] }), _jsxs("div", { className: "pt-2 border-t border-gray-700/50", children: [_jsx("label", { className: "block text-[10px] font-bold text-gray-500 uppercase mb-2", children: "Array Specs" }), _jsxs("div", { className: "space-y-2", children: [_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-[10px] text-gray-400", children: [_jsx("span", { children: "Efficiency" }), _jsxs("span", { children: [(panelEfficiency * 100).toFixed(0), "%"] })] }), _jsx("input", { type: "range", min: "0.1", max: "0.5", step: "0.01", value: panelEfficiency, onChange: (e) => setPanelEfficiency(parseFloat(e.target.value)), className: "w-full accent-amber-500 h-1.5" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-[10px] text-gray-400", children: [_jsx("span", { children: "Area" }), _jsxs("span", { children: [panelArea.toFixed(0), " m\u00B2"] })] }), _jsx("input", { type: "range", min: "10", max: "500", step: "10", value: panelArea, onChange: (e) => setPanelArea(parseFloat(e.target.value)), className: "w-full accent-amber-500 h-1.5" })] })] })] }), _jsxs("div", { className: "pt-2 border-t border-gray-700/50", children: [_jsx("label", { className: "block text-[10px] font-bold text-gray-500 uppercase mb-2", children: "Mission Specs" }), _jsxs("div", { className: "space-y-2", children: [_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-[10px] text-gray-400", children: [_jsx("span", { children: "Battery" }), _jsxs("span", { children: [batteryCapacity.toFixed(0), " kWh"] })] }), _jsx("input", { type: "range", min: "10", max: "200", step: "10", value: batteryCapacity, onChange: (e) => setBatteryCapacity(parseFloat(e.target.value)), className: "w-full accent-amber-500 h-1.5" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-[10px] text-gray-400", children: [_jsx("span", { children: "Daily Load" }), _jsxs("span", { children: [dailyPowerNeeds.toFixed(1), " kWh"] })] }), _jsx("input", { type: "range", min: "5", max: "50", step: "1", value: dailyPowerNeeds, onChange: (e) => setDailyPowerNeeds(parseFloat(e.target.value)), className: "w-full accent-amber-500 h-1.5" })] })] })] })] })] }), results && (_jsxs("div", { className: "glass-panel p-4 rounded-lg border border-amber-500/20", children: [_jsx("h3", { className: "text-xs font-bold mb-3 text-amber-400 tracking-wider uppercase", children: "Solar Metrics" }), _jsxs("div", { className: "grid grid-cols-2 gap-2 mb-4", children: [_jsxs("div", { className: "bg-gray-900 p-2 rounded border border-gray-700", children: [_jsx("div", { className: "text-[10px] text-gray-500 uppercase", children: "Mean Potential" }), _jsxs("div", { className: "text-lg font-bold text-white", children: [(results.statistics.mean * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "bg-gray-900 p-2 rounded border border-gray-700", children: [_jsx("div", { className: "text-[10px] text-gray-500 uppercase", children: "Irradiance" }), _jsxs("div", { className: "text-lg font-bold text-amber-400", children: [results.statistics.mean_irradiance_w_per_m2.toFixed(0), " ", _jsx("span", { className: "text-xs text-gray-500 font-normal", children: "W/m\u00B2" })] })] })] }), _jsxs("div", { className: "text-xs text-gray-300 space-y-1", children: [_jsxs("div", { className: "flex justify-between", children: [_jsx("span", { children: "Power Gen (Daily):" }), _jsxs("span", { className: "font-mono text-white", children: [results.mission_impacts.power_generation_kwh_per_day.toFixed(1), " kWh"] })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { children: "Surplus:" }), _jsxs("span", { className: `font-mono ${results.mission_impacts.power_surplus_kwh_per_day >= 0 ? 'text-green-400' : 'text-red-400'}`, children: [results.mission_impacts.power_surplus_kwh_per_day.toFixed(1), " kWh"] })] })] })] }))] }));
}
