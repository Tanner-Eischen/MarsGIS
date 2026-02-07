import { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import { apiFetch, apiUrl } from '../lib/apiBase';
// Custom hook to fetch and manage the DEM image
export function useDemImage(roi, dataset, relief) {
    const [demImage, setDemImage] = useState({
        url: null,
        bounds: null,
        loading: false,
        error: null,
    });
    const imageUrlRef = useRef(null);
    useEffect(() => {
        if (!roi)
            return;
        const fetchDemImage = async () => {
            setDemImage((prev) => ({ ...prev, loading: true, error: null }));
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
                imageUrlRef.current = null;
            }
            try {
                const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`;
                const url = apiUrl(`/visualization/dem-image?dataset=${dataset}&roi=${roiStr}&width=2400&height=1600&colormap=terrain&relief=${relief}&buffer=1.5`);
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000);
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Failed to load DEM image: ${response.status} ${errorText}`);
                }
                const blob = await response.blob();
                if (blob.size === 0) {
                    throw new Error('Empty DEM image received');
                }
                const newUrl = URL.createObjectURL(blob);
                imageUrlRef.current = newUrl;
                const left = parseFloat(response.headers.get('X-Bounds-Left') || String(roi.lon_min));
                const right = parseFloat(response.headers.get('X-Bounds-Right') || String(roi.lon_max));
                const bottom = parseFloat(response.headers.get('X-Bounds-Bottom') || String(roi.lat_min));
                const top = parseFloat(response.headers.get('X-Bounds-Top') || String(roi.lat_max));
                const bounds = L.latLngBounds([bottom, left], [top, right]);
                if (!bounds.isValid()) {
                    const fallbackBounds = L.latLngBounds([roi.lat_min, roi.lon_min], [roi.lat_max, roi.lon_max]);
                    setDemImage({ url: newUrl, bounds: fallbackBounds, loading: false, error: null });
                }
                else {
                    setDemImage({ url: newUrl, bounds, loading: false, error: null });
                }
            }
            catch (error) {
                if (error instanceof Error && error.name === 'AbortError') {
                    setDemImage({ url: null, bounds: null, loading: false, error: 'DEM loading timed out.' });
                }
                else {
                    const message = error instanceof Error ? error.message : 'Unknown error';
                    setDemImage({ url: null, bounds: null, loading: false, error: message });
                }
            }
        };
        fetchDemImage();
        return () => {
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
            }
        };
    }, [roi, dataset, relief]);
    return demImage;
}
function useGeoJson(path, enabled) {
    const [data, setData] = useState(null);
    useEffect(() => {
        if (!enabled) {
            setData(null);
            return;
        }
        const fetchData = async () => {
            try {
                const response = await apiFetch(path);
                if (!response.ok) {
                    return;
                }
                const geojsonData = await response.json();
                setData(geojsonData);
            }
            catch {
                // Best-effort overlay fetch; map UI handles missing data.
            }
        };
        fetchData();
    }, [path, enabled]);
    return data;
}
export function useSitesGeoJson(enabled) {
    return useGeoJson('/visualization/sites-geojson', enabled);
}
export function useWaypointsGeoJson(enabled) {
    return useGeoJson('/visualization/waypoints-geojson', enabled);
}
// Unified overlay hook
// Note: For caching support, use OverlayLayerContext directly in components
export function useOverlayImage(roi, dataset, overlayType, options = {}) {
    const [overlayImage, setOverlayImage] = useState({
        url: null,
        bounds: null,
        loading: false,
        error: null,
    });
    const imageUrlRef = useRef(null);
    const { colormap = 'terrain', relief = 0, sunAzimuth = 315, sunAltitude = 45, width = 2400, height = 1600, buffer = 1.5, marsSol, season, dustStormPeriod, } = options;
    useEffect(() => {
        if (!roi || !overlayType)
            return;
        const fetchOverlayImage = async () => {
            setOverlayImage((prev) => ({ ...prev, loading: true, error: null }));
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
                imageUrlRef.current = null;
            }
            try {
                const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`;
                const params = new URLSearchParams({
                    overlay_type: overlayType,
                    dataset,
                    roi: roiStr,
                    colormap,
                    relief: String(relief),
                    sun_azimuth: String(sunAzimuth),
                    sun_altitude: String(sunAltitude),
                    width: String(width),
                    height: String(height),
                    buffer: String(buffer),
                });
                if (marsSol !== undefined)
                    params.append('mars_sol', String(marsSol));
                if (season)
                    params.append('season', season);
                if (dustStormPeriod)
                    params.append('dust_storm_period', dustStormPeriod);
                const url = apiUrl(`/visualization/overlay?${params.toString()}`);
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 60000);
                const response = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`Failed to load overlay image: ${response.status} ${errorText}`);
                }
                const blob = await response.blob();
                if (blob.size === 0) {
                    throw new Error('Empty overlay image received');
                }
                const newUrl = URL.createObjectURL(blob);
                imageUrlRef.current = newUrl;
                const left = parseFloat(response.headers.get('X-Bounds-Left') || String(roi.lon_min));
                const right = parseFloat(response.headers.get('X-Bounds-Right') || String(roi.lon_max));
                const bottom = parseFloat(response.headers.get('X-Bounds-Bottom') || String(roi.lat_min));
                const top = parseFloat(response.headers.get('X-Bounds-Top') || String(roi.lat_max));
                const bounds = L.latLngBounds([bottom, left], [top, right]);
                if (!bounds.isValid()) {
                    const fallbackBounds = L.latLngBounds([roi.lat_min, roi.lon_min], [roi.lat_max, roi.lon_max]);
                    setOverlayImage({ url: newUrl, bounds: fallbackBounds, loading: false, error: null });
                }
                else {
                    setOverlayImage({ url: newUrl, bounds, loading: false, error: null });
                }
            }
            catch (error) {
                if (error instanceof Error && error.name === 'AbortError') {
                    setOverlayImage({
                        url: null,
                        bounds: null,
                        loading: false,
                        error: 'Overlay loading timed out.',
                    });
                }
                else {
                    const message = error instanceof Error ? error.message : 'Unknown error';
                    setOverlayImage({ url: null, bounds: null, loading: false, error: message });
                }
            }
        };
        fetchOverlayImage();
        return () => {
            if (imageUrlRef.current) {
                URL.revokeObjectURL(imageUrlRef.current);
            }
        };
    }, [
        roi,
        dataset,
        overlayType,
        colormap,
        relief,
        sunAzimuth,
        sunAltitude,
        width,
        height,
        buffer,
        marsSol,
        season,
        dustStormPeriod,
    ]);
    return overlayImage;
}
