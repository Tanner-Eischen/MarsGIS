import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import SolarHeatmap from '../components/SolarHeatmap';
import SolarImpactPanel from '../components/SolarImpactPanel';
import { analyzeSolarPotential, getExampleROIs } from '../services/api';
export default function SolarAnalysis() {
    // Default to Jezero Crater
    const [roi, setROI] = useState({
        lat_min: 18.0,
        lat_max: 18.6,
        lon_min: 77.0,
        lon_max: 77.8,
    });
    const [dataset, setDataset] = useState('mola');
    const [examples, setExamples] = useState([]);
    useEffect(() => {
        const s = localStorage.getItem('solar.roi');
        const d = localStorage.getItem('solar.dataset');
        if (s) {
            try {
                const o = JSON.parse(s);
                if (o && o.lat_min !== undefined)
                    setROI(o);
            }
            catch { }
        }
        if (d)
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
    return (_jsxs("div", { className: "space-y-6", children: [_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("h2", { className: "text-3xl font-bold", children: "Solar Potential Analysis" }), _jsx("button", { onClick: handleAnalyze, disabled: loading, className: "px-6 py-2 bg-mars-orange hover:bg-orange-600 text-white rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed", children: loading ? 'Analyzing...' : 'Analyze Solar Potential' })] }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("button", { onClick: async () => { const data = await getExampleROIs(); setExamples(data); }, className: "px-3 py-2 bg-gray-700 text-white rounded", children: "Fill with Example ROI" }), examples.length > 0 && (_jsxs("select", { onChange: (e) => {
                            const sel = examples.find(x => x.id === e.target.value);
                            if (sel) {
                                setROI(sel.bbox);
                                setDataset(sel.dataset);
                            }
                        }, className: "bg-gray-700 text-white px-2 py-2 rounded", defaultValue: "", children: [_jsx("option", { value: "", disabled: true, children: "Select example" }), examples.map(x => (_jsx("option", { value: x.id, children: x.name }, x.id)))] }))] }), error && (_jsxs("div", { className: "p-4 bg-red-900/50 border border-red-700 rounded-md", children: [_jsx("p", { className: "text-red-300 font-semibold", children: "Error" }), _jsx("p", { className: "text-red-400 text-sm mt-1", children: error })] })), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-3 gap-6", children: [_jsxs("div", { className: "lg:col-span-1 space-y-6", children: [_jsxs("div", { className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Region of Interest" }), _jsxs("div", { className: "grid grid-cols-2 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Min" }), _jsx("input", { type: "number", value: isNaN(roi.lat_min) ? '' : roi.lat_min, onChange: (e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val))
                                                                setROI({ ...roi, lat_min: val });
                                                        }, className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Latitude Max" }), _jsx("input", { type: "number", value: isNaN(roi.lat_max) ? '' : roi.lat_max, onChange: (e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val))
                                                                setROI({ ...roi, lat_max: val });
                                                        }, className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Min" }), _jsx("input", { type: "number", value: isNaN(roi.lon_min) ? '' : roi.lon_min, onChange: (e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val))
                                                                setROI({ ...roi, lon_min: val });
                                                        }, className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Longitude Max" }), _jsx("input", { type: "number", value: isNaN(roi.lon_max) ? '' : roi.lon_max, onChange: (e) => {
                                                            const val = parseFloat(e.target.value);
                                                            if (!isNaN(val))
                                                                setROI({ ...roi, lon_max: val });
                                                        }, className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", step: "0.1" })] })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium mb-2", children: "Dataset" }), _jsxs("select", { value: dataset, onChange: (e) => setDataset(e.target.value), className: "w-full bg-gray-700 text-white px-4 py-2 rounded-md", children: [_jsx("option", { value: "mola", children: "MOLA" }), _jsx("option", { value: "hirise", children: "HiRISE" }), _jsx("option", { value: "ctx", children: "CTX" })] })] })] }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Solar Panel Configuration" }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Panel Efficiency: ", (panelEfficiency * 100).toFixed(0), "%"] }), _jsx("input", { type: "range", min: "0.1", max: "0.5", step: "0.01", value: panelEfficiency, onChange: (e) => setPanelEfficiency(parseFloat(e.target.value)), className: "w-full" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Panel Area: ", panelArea.toFixed(0), " m\u00B2"] }), _jsx("input", { type: "range", min: "10", max: "500", step: "10", value: panelArea, onChange: (e) => setPanelArea(parseFloat(e.target.value)), className: "w-full" })] })] }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Mission Parameters" }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Battery Capacity: ", batteryCapacity.toFixed(0), " kWh"] }), _jsx("input", { type: "range", min: "10", max: "200", step: "10", value: batteryCapacity, onChange: (e) => setBatteryCapacity(parseFloat(e.target.value)), className: "w-full" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Daily Power Needs: ", dailyPowerNeeds.toFixed(1), " kWh/day"] }), _jsx("input", { type: "range", min: "5", max: "50", step: "1", value: dailyPowerNeeds, onChange: (e) => setDailyPowerNeeds(parseFloat(e.target.value)), className: "w-full" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Mission Duration: ", missionDuration.toFixed(0), " days"] }), _jsx("input", { type: "range", min: "100", max: "1000", step: "50", value: missionDuration, onChange: (e) => setMissionDuration(parseFloat(e.target.value)), className: "w-full" })] })] }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 space-y-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Sun Position" }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Sun Azimuth: ", sunAzimuth.toFixed(0), "\u00B0 (0\u00B0 = North)"] }), _jsx("input", { type: "range", min: "0", max: "360", step: "15", value: sunAzimuth, onChange: (e) => setSunAzimuth(parseFloat(e.target.value)), className: "w-full" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium mb-2", children: ["Sun Altitude: ", sunAltitude.toFixed(0), "\u00B0 (0\u00B0 = Horizon)"] }), _jsx("input", { type: "range", min: "0", max: "90", step: "5", value: sunAltitude, onChange: (e) => setSunAltitude(parseFloat(e.target.value)), className: "w-full" })] })] })] }), _jsxs("div", { className: "lg:col-span-2 space-y-6", children: [results && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsx("h3", { className: "text-xl font-semibold mb-4", children: "Solar Potential Statistics" }), _jsxs("div", { className: "grid grid-cols-3 gap-4", children: [_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("div", { className: "text-sm text-gray-400", children: "Mean Potential" }), _jsxs("div", { className: "text-2xl font-bold text-white", children: [(results.statistics.mean * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("div", { className: "text-sm text-gray-400", children: "Max Potential" }), _jsxs("div", { className: "text-2xl font-bold text-white", children: [(results.statistics.max * 100).toFixed(1), "%"] })] }), _jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("div", { className: "text-sm text-gray-400", children: "Mean Irradiance" }), _jsxs("div", { className: "text-2xl font-bold text-white", children: [results.statistics.mean_irradiance_w_per_m2.toFixed(0), " ", _jsx("span", { className: "text-sm", children: "W/m\u00B2" })] })] })] })] }), _jsx(SolarHeatmap, { roi: roi, dataset: dataset, solarPotentialMap: results.solar_potential_map, shape: results.shape, loading: loading }), _jsx(SolarImpactPanel, { impacts: results.mission_impacts, dailyPowerNeeds: dailyPowerNeeds })] })), !results && !loading && (_jsx("div", { className: "bg-gray-800 rounded-lg p-12 text-center", children: _jsx("p", { className: "text-gray-400 text-lg", children: "Configure parameters and click \"Analyze Solar Potential\" to begin" }) }))] })] })] }));
}
