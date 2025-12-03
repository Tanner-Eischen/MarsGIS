import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { MapContainer, ImageOverlay, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useDemImage, useSitesGeoJson, useWaypointsGeoJson } from '../hooks/useMapData';
import { useEffect } from 'react';
// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});
// --- Helper Components ---
function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => {
        if (bounds) {
            map.fitBounds(bounds, { padding: [0, 0], maxZoom: 12 });
        }
    }, [map, bounds]);
    return null;
}
// --- Feature Styling and Interaction ---
const siteStyle = (isSelected) => ({
    radius: isSelected ? 12 : 8,
    fillColor: isSelected ? '#ffff00' : '#00ff00',
    color: isSelected ? '#ff0000' : '#ffffff',
    weight: isSelected ? 3 : 2,
    opacity: 1,
    fillOpacity: 0.8,
});
const waypointStyle = {
    radius: 3,
    fillColor: '#ff0000',
    color: '#ffffff',
    weight: 1.5,
    opacity: 1,
    fillOpacity: 0.8,
};
const pointToLayer = (feature, latlng, selectedSiteId) => {
    const isWaypoint = !!feature.properties.waypoint_id;
    if (isWaypoint) {
        return L.circleMarker(latlng, waypointStyle);
    }
    const siteId = feature.properties?.site_id;
    const isSelected = selectedSiteId === siteId;
    return L.circleMarker(latlng, siteStyle(isSelected));
};
const onEachFeature = (feature, layer, onSiteSelect) => {
    const props = feature.properties;
    if (props && props.site_id) {
        const popupContent = `
      <div>
        <strong>Site ${props.site_id}</strong><br/>
        Rank: ${props.rank}<br/>
        Area: ${props.area_km2.toFixed(2)} km²<br/>
        Suitability: ${(props.suitability_score * 100).toFixed(1)}%<br/>
        Slope: ${props.mean_slope_deg.toFixed(2)}°<br/>
        Elevation: ${props.mean_elevation_m.toFixed(0)} m
      </div>
    `;
        layer.bindPopup(popupContent);
        if (onSiteSelect) {
            layer.on('click', () => onSiteSelect(props.site_id));
        }
    }
};
export default function TerrainMap({ roi, dataset = 'mola', showSites = false, showWaypoints = false, relief = 0, onSiteSelect, selectedSiteId }) {
    const { url: demImageUrl, bounds: imageBounds, loading, error } = useDemImage(roi, dataset, relief);
    const sitesGeoJson = useSitesGeoJson(showSites);
    const waypointsGeoJson = useWaypointsGeoJson(showWaypoints);
    const centerTuple = roi
        ? [(roi.lat_min + roi.lat_max) / 2, (roi.lon_min + roi.lon_max) / 2]
        : [0, 180];
    if (loading) {
        return _jsx("div", { className: "map-loading", children: "Loading Map..." });
    }
    if (error) {
        return _jsxs("div", { className: "map-error", children: ["Error: ", error] });
    }
    return (_jsxs(MapContainer, { center: centerTuple, zoom: 6, style: { height: '100%', width: '100%' }, children: [demImageUrl && imageBounds && (_jsxs(_Fragment, { children: [_jsx(ImageOverlay, { url: demImageUrl, bounds: imageBounds }), _jsx(FitBounds, { bounds: imageBounds })] })), sitesGeoJson && (_jsx(GeoJSON, { data: sitesGeoJson, pointToLayer: (f, l) => pointToLayer(f, l, selectedSiteId), onEachFeature: (f, l) => onEachFeature(f, l, onSiteSelect) })), waypointsGeoJson && (_jsx(GeoJSON, { data: waypointsGeoJson, pointToLayer: (f, l) => pointToLayer(f, l, selectedSiteId), style: (feature) => ({ color: feature?.properties?.line_color || '#ff0000', weight: 2, opacity: 0.8 }) }))] }));
}
