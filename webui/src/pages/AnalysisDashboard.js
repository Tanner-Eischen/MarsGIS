import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import TerrainMap from '../components/TerrainMap';
import Terrain3D from '../components/Terrain3D';
import OverlaySwitcher from '../components/OverlaySwitcher';
import TerrainAnalysis from './TerrainAnalysis';
import SolarAnalysis from './SolarAnalysis';
import { Layers, Sun, Box } from 'lucide-react';
export default function AnalysisDashboard() {
    const [mode, setMode] = useState('terrain');
    const [sidebarOpen, setSidebarOpen] = useState(true);
    // Shared Map State
    const [roi, setRoi] = useState({ lat_min: 18.0, lat_max: 18.6, lon_min: 77.0, lon_max: 77.8 });
    const [dataset, setDataset] = useState('mola');
    const [overlayType, setOverlayType] = useState('elevation');
    const [colormap, setColormap] = useState('terrain');
    const [relief, setRelief] = useState(1.0);
    const [sunAzimuth, setSunAzimuth] = useState(315);
    const [sunAltitude, setSunAltitude] = useState(45);
    // Sync overlay type with mode
    useEffect(() => {
        if (mode === 'solar' && overlayType !== 'solar') {
            setOverlayType('solar');
            setColormap('plasma');
        }
        else if (mode === 'terrain' && overlayType === 'solar') {
            setOverlayType('elevation');
            setColormap('terrain');
        }
    }, [mode]);
    return (_jsxs("div", { className: "flex h-[calc(100vh-4rem)] overflow-hidden bg-gray-900 text-white", children: [_jsxs("div", { className: `flex flex-col bg-gray-800/90 backdrop-blur border-r border-gray-700 transition-all duration-300 ${sidebarOpen ? 'w-96' : 'w-16'}`, children: [_jsxs("div", { className: "p-4 border-b border-gray-700 flex items-center justify-between", children: [sidebarOpen && _jsx("h2", { className: "font-bold text-lg tracking-wider text-cyan-400", children: "ANALYSIS_HUB" }), _jsx("button", { onClick: () => setSidebarOpen(!sidebarOpen), className: "p-2 hover:bg-gray-700 rounded text-cyan-500", children: sidebarOpen ? '<<' : '>>' })] }), _jsxs("div", { className: "flex flex-col p-2 gap-2", children: [_jsxs("button", { onClick: () => setMode('terrain'), className: `flex items-center p-3 rounded transition-all ${mode === 'terrain' ? 'bg-cyan-900/50 text-cyan-300 border border-cyan-500/50' : 'hover:bg-gray-700 text-gray-400'}`, title: "Terrain Analysis", children: [_jsx(Layers, { size: 20 }), sidebarOpen && _jsx("span", { className: "ml-3 font-medium", children: "Terrain Analysis" })] }), _jsxs("button", { onClick: () => setMode('solar'), className: `flex items-center p-3 rounded transition-all ${mode === 'solar' ? 'bg-amber-900/50 text-amber-300 border border-amber-500/50' : 'hover:bg-gray-700 text-gray-400'}`, title: "Solar Potential", children: [_jsx(Sun, { size: 20 }), sidebarOpen && _jsx("span", { className: "ml-3 font-medium", children: "Solar Potential" })] }), _jsxs("button", { onClick: () => setMode('3d'), className: `flex items-center p-3 rounded transition-all ${mode === '3d' ? 'bg-purple-900/50 text-purple-300 border border-purple-500/50' : 'hover:bg-gray-700 text-gray-400'}`, title: "3D Visualization", children: [_jsx(Box, { size: 20 }), sidebarOpen && _jsx("span", { className: "ml-3 font-medium", children: "3D Visualization" })] })] }), sidebarOpen && (_jsxs("div", { className: "flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar", children: [_jsx("div", { className: "bg-gray-900/50 p-3 rounded border border-gray-700/50", children: _jsx(OverlaySwitcher, { overlayType: overlayType, onOverlayTypeChange: setOverlayType, colormap: colormap, onColormapChange: setColormap, relief: relief, onReliefChange: setRelief, sunAzimuth: sunAzimuth, onSunAzimuthChange: setSunAzimuth, sunAltitude: sunAltitude, onSunAltitudeChange: setSunAltitude, dataset: dataset, roi: roi, showLayerList: false, showCacheStats: false }) }), _jsxs("div", { className: "space-y-4", children: [mode === 'terrain' && (_jsxs("div", { className: "text-sm text-gray-400", children: [_jsx("h3", { className: "text-cyan-400 font-semibold mb-2 uppercase tracking-wider", children: "Terrain Parameters" }), _jsx(TerrainAnalysis, { roi: roi, onRoiChange: setRoi, dataset: dataset, onDatasetChange: setDataset })] })), mode === 'solar' && (_jsxs("div", { className: "text-sm text-gray-400", children: [_jsx("h3", { className: "text-amber-400 font-semibold mb-2 uppercase tracking-wider", children: "Solar Config" }), _jsx(SolarAnalysis, { roi: roi, onRoiChange: setRoi, dataset: dataset, onDatasetChange: setDataset })] }))] })] }))] }), _jsxs("div", { className: "flex-1 relative bg-black", children: [mode === '3d' ? (_jsx(Terrain3D, {})) : (_jsx(TerrainMap, { roi: roi, dataset: dataset, overlayType: overlayType, overlayOptions: {
                            colormap,
                            relief,
                            sunAzimuth,
                            sunAltitude
                        }, showSites: true, showWaypoints: true })), _jsxs("div", { className: "absolute top-4 right-4 bg-gray-900/80 backdrop-blur border border-cyan-500/30 p-2 rounded text-xs font-mono text-cyan-400 pointer-events-none", children: [_jsxs("div", { children: ["LAT: ", (roi.lat_min + roi.lat_max) / 2, "\u00B0"] }), _jsxs("div", { children: ["LON: ", (roi.lon_min + roi.lon_max) / 2, "\u00B0"] }), _jsx("div", { children: "ZOOM: 100%" })] })] })] }));
}
