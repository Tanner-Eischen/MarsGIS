import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from 'react';
import { useOverlayLayerManager } from '../hooks/useOverlayLayerManager';
const OverlayLayerContext = createContext(undefined);
export function OverlayLayerProvider({ children, maxCacheSize = 5 }) {
    const layerManager = useOverlayLayerManager({ maxCacheSize });
    const value = {
        layers: layerManager.layers,
        activeLayerName: layerManager.activeLayerName,
        loadLayer: layerManager.loadLayer,
        switchLayer: layerManager.switchLayer,
        preloadAllLayers: layerManager.preloadAllLayers,
        clearCache: layerManager.clearCache,
        getCacheStats: layerManager.getCacheStats
    };
    return (_jsx(OverlayLayerContext.Provider, { value: value, children: children }));
}
export function useOverlayLayerContext() {
    const context = useContext(OverlayLayerContext);
    if (context === undefined) {
        throw new Error('useOverlayLayerContext must be used within an OverlayLayerProvider');
    }
    return context;
}
