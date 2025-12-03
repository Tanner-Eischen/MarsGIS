import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext, useMemo, useState } from 'react';
const GeoPlanContext = createContext(null);
export function GeoPlanProvider({ children }) {
    const [landingSites, setLandingSites] = useState([]);
    const [constructionSites, setConstructionSites] = useState([]);
    const [recommendedLandingSiteId, setRecommendedLandingSiteId] = useState(null);
    const publishSites = (type, sites) => {
        if (type === 'landing')
            setLandingSites(sites);
        else
            setConstructionSites(sites);
    };
    const value = useMemo(() => ({
        landingSites,
        constructionSites,
        recommendedLandingSiteId,
        publishSites,
        setRecommendedLandingSiteId,
    }), [landingSites, constructionSites, recommendedLandingSiteId]);
    return _jsx(GeoPlanContext.Provider, { value: value, children: children });
}
export function useGeoPlan() {
    const ctx = useContext(GeoPlanContext);
    if (!ctx)
        throw new Error('GeoPlanContext not initialized');
    return ctx;
}
