import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { useOverlayLayerContext } from '../contexts/OverlayLayerContext';
import { getOverlayDefinition, MARS_OVERLAY_DEFINITIONS } from '../config/marsDataSources';
import LayerStatusBadge from './LayerStatusBadge';
import { Layers, Eye, EyeOff, Database, Trash2, Settings, Sun } from 'lucide-react';
const COLORMAPS = [
    { value: 'terrain', label: 'Natural Terrain' },
    { value: 'viridis', label: 'Viridis (Analytical)' },
    { value: 'plasma', label: 'Plasma (Solar/Heat)' },
    { value: 'magma', label: 'Magma (Geological)' },
    { value: 'cividis', label: 'Cividis (Colorblind Safe)' },
];
export default function OverlaySwitcher({ overlayType, onOverlayTypeChange, colormap, onColormapChange, relief, onReliefChange, sunAzimuth, onSunAzimuthChange, sunAltitude, onSunAltitudeChange, dataset = 'mola', roi, showLayerList = true, showCacheStats = true, }) {
    const [expanded, setExpanded] = useState(true);
    const [activeTab, setActiveTab] = useState('layers');
    const overlayContext = useOverlayLayerContext();
    const cacheStats = overlayContext.getCacheStats();
    const showColormap = overlayType !== 'hillshade';
    const showRelief = overlayType === 'elevation';
    const showSunAngles = overlayType === 'solar' || overlayType === 'hillshade' || overlayType === 'dust';
    const activeOverlayDef = getOverlayDefinition(overlayType);
    const handlePreloadAll = async () => {
        if (!roi)
            return;
        try {
            await overlayContext.preloadAllLayers(dataset, roi, {
                colormap,
                relief,
                sunAzimuth,
                sunAltitude
            });
        }
        catch (error) {
            console.error('Failed to preload layers:', error);
        }
    };
    const handleClearCache = () => {
        overlayContext.clearCache();
    };
    const getLayerStatus = (layerName) => {
        const layer = overlayContext.layers.find(l => l.name === layerName);
        if (!layer)
            return 'idle';
        if (layer.loading)
            return 'loading';
        if (layer.loaded && layer.data)
            return 'cached';
        if (layer.error)
            return 'error';
        return 'idle';
    };
    return (_jsxs("div", { className: "glass-panel rounded-lg overflow-hidden text-sm", children: [_jsxs("div", { className: "bg-gray-800/50 p-3 border-b border-gray-700 flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center gap-2 text-cyan-400", children: [_jsx(Layers, { size: 16 }), _jsx("h3", { className: "font-bold tracking-wider", children: "VISUAL_LAYERS" })] }), _jsxs("div", { className: "flex gap-2", children: [_jsx("button", { onClick: () => setActiveTab('layers'), className: `p-1 rounded ${activeTab === 'layers' ? 'bg-cyan-900/50 text-cyan-300' : 'text-gray-500 hover:text-gray-300'}`, children: _jsx(Layers, { size: 14 }) }), _jsx("button", { onClick: () => setActiveTab('settings'), className: `p-1 rounded ${activeTab === 'settings' ? 'bg-cyan-900/50 text-cyan-300' : 'text-gray-500 hover:text-gray-300'}`, children: _jsx(Settings, { size: 14 }) }), _jsx("button", { onClick: () => setExpanded(!expanded), className: "text-gray-500 hover:text-white p-1", children: expanded ? _jsx(Eye, { size: 14 }) : _jsx(EyeOff, { size: 14 }) })] })] }), expanded && (_jsxs("div", { className: "p-3 space-y-4 bg-gray-900/30", children: [activeTab === 'layers' && (_jsx("div", { className: "grid grid-cols-2 gap-2", children: MARS_OVERLAY_DEFINITIONS.filter(def => ['elevation', 'solar', 'viewshed', 'comms_risk', 'hillshade', 'slope', 'roughness', 'tri'].includes(def.name)).map((def) => {
                            const isActive = overlayType === def.name;
                            const status = getLayerStatus(def.name);
                            return (_jsxs("button", { onClick: () => onOverlayTypeChange(def.name), className: `relative flex flex-col items-start p-2 rounded border transition-all duration-200 ${isActive
                                    ? 'bg-cyan-900/20 border-cyan-500/50 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.1)]'
                                    : 'bg-gray-800/50 border-gray-700 text-gray-400 hover:bg-gray-700 hover:border-gray-500'}`, children: [_jsxs("div", { className: "flex items-center justify-between w-full mb-1", children: [_jsx("span", { className: "font-mono text-xs font-bold uppercase", children: def.displayName }), _jsx(LayerStatusBadge, { status: status, dataset: dataset })] }), _jsx("span", { className: "text-[10px] opacity-70 truncate w-full text-left", children: def.description })] }, def.name));
                        }) })), activeTab === 'settings' && (_jsxs("div", { className: "space-y-4", children: [activeOverlayDef && (_jsx("div", { className: "p-2 bg-cyan-900/10 border border-cyan-500/20 rounded text-xs text-cyan-200/80 font-mono", children: _jsxs("div", { className: "flex justify-between", children: [_jsxs("span", { children: ["RES: ", activeOverlayDef.resolution || 'N/A'] }), _jsxs("span", { children: ["SRC: ", dataset.toUpperCase()] })] }) })), showColormap && (_jsxs("div", { children: [_jsx("label", { className: "block text-xs font-bold text-gray-400 mb-1 uppercase", children: "Spectral Palette" }), _jsx("select", { value: colormap, onChange: (e) => onColormapChange(e.target.value), className: "w-full bg-gray-800 border border-gray-600 text-white px-2 py-1 rounded text-xs focus:border-cyan-500 focus:outline-none", children: COLORMAPS.map((cm) => (_jsx("option", { value: cm.value, children: cm.label }, cm.value))) })] })), showRelief && (_jsxs("div", { children: [_jsxs("label", { className: "block text-xs font-bold text-gray-400 mb-1 uppercase", children: ["Exaggeration: ", relief.toFixed(1), "x"] }), _jsx("input", { type: "range", min: "0", max: "3", step: "0.1", value: relief, onChange: (e) => onReliefChange(parseFloat(e.target.value)), className: "w-full" })] })), showSunAngles && (_jsxs("div", { className: "space-y-2 pt-2 border-t border-gray-700/50", children: [_jsxs("div", { className: "flex items-center gap-2 text-amber-500 mb-1", children: [_jsx(Sun, { size: 12 }), _jsx("span", { className: "text-xs font-bold uppercase", children: "Solar Ephemeris" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-xs text-gray-400", children: [_jsx("span", { children: "Azimuth" }), _jsxs("span", { children: [sunAzimuth.toFixed(0), "\u00B0"] })] }), _jsx("input", { type: "range", min: "0", max: "360", step: "15", value: sunAzimuth, onChange: (e) => onSunAzimuthChange(parseFloat(e.target.value)), className: "w-full accent-amber-500" })] }), _jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-xs text-gray-400", children: [_jsx("span", { children: "Altitude" }), _jsxs("span", { children: [sunAltitude.toFixed(0), "\u00B0"] })] }), _jsx("input", { type: "range", min: "0", max: "90", step: "5", value: sunAltitude, onChange: (e) => onSunAltitudeChange(parseFloat(e.target.value)), className: "w-full accent-amber-500" })] })] }))] })), showCacheStats && (_jsxs("div", { className: "pt-2 border-t border-gray-700/50", children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("span", { className: "text-xs font-bold text-gray-500 uppercase", children: "Memory Cache" }), _jsxs("span", { className: "text-xs font-mono text-cyan-400", children: [(cacheStats.memoryUsageKB / 1024).toFixed(1), " MB"] })] }), _jsxs("div", { className: "grid grid-cols-2 gap-2", children: [_jsxs("button", { onClick: handlePreloadAll, disabled: !roi, className: "flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-800 hover:bg-gray-700 text-xs rounded border border-gray-600 transition-colors disabled:opacity-50", children: [_jsx(Database, { size: 10 }), "PRELOAD"] }), _jsxs("button", { onClick: handleClearCache, className: "flex items-center justify-center gap-1 px-2 py-1.5 bg-gray-800 hover:bg-red-900/30 text-xs rounded border border-gray-600 hover:border-red-500/50 transition-colors", children: [_jsx(Trash2, { size: 10 }), "PURGE"] })] })] }))] }))] }));
}
