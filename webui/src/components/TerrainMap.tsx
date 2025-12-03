import { MapContainer, ImageOverlay, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useDemImage, useSitesGeoJson, useWaypointsGeoJson } from '../hooks/useMapData';
import { useEffect } from 'react';

// Fix for default marker icons
delete (L.Icon.Default.prototype as any)._getIconUrl;
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

// --- Main Component ---

interface TerrainMapProps {
  roi: any
  dataset?: string
  showSites?: boolean
  showWaypoints?: boolean
  relief?: number
  onSiteSelect?: any
  selectedSiteId?: any
  overlayType?: 'elevation' | 'solar' | 'dust' | 'hillshade' | 'slope' | 'aspect' | 'roughness' | 'tri'
  overlayOptions?: {
    colormap?: string
    relief?: number
    sunAzimuth?: number
    sunAltitude?: number
    width?: number
    height?: number
    buffer?: number
    marsSol?: number
    season?: string
    dustStormPeriod?: string
  }
}

export default function TerrainMap({ 
  roi, 
  dataset = 'mola', 
  showSites = false, 
  showWaypoints = false, 
  relief = 0, 
  onSiteSelect, 
  selectedSiteId,
  overlayType,
  overlayOptions = {}
}: TerrainMapProps) {
  // Use overlay system if overlayType is provided, otherwise use DEM
  const useOverlay = overlayType && overlayType !== 'elevation'
  
  const { useOverlayImage } = require('../hooks/useMapData')
  const overlayImage = useOverlay 
    ? useOverlayImage(roi, dataset, overlayType, {
        colormap: overlayOptions.colormap || 'terrain',
        relief: overlayOptions.relief ?? relief,
        sunAzimuth: overlayOptions.sunAzimuth || 315,
        sunAltitude: overlayOptions.sunAltitude || 45,
        width: overlayOptions.width || 2400,
        height: overlayOptions.height || 1600,
        buffer: overlayOptions.buffer || 1.5,
        marsSol: overlayOptions.marsSol,
        season: overlayOptions.season,
        dustStormPeriod: overlayOptions.dustStormPeriod
      })
    : null

  const { url: demImageUrl, bounds: imageBounds, loading, error } = useDemImage(roi, dataset, relief);
  const sitesGeoJson = useSitesGeoJson(showSites);
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints);

  // Use overlay image if available, otherwise fall back to DEM
  const displayImageUrl = useOverlay && overlayImage?.url ? overlayImage.url : demImageUrl
  const displayBounds = useOverlay && overlayImage?.bounds ? overlayImage.bounds : imageBounds
  const displayLoading = useOverlay && overlayImage ? overlayImage.loading : loading
  const displayError = useOverlay && overlayImage ? overlayImage.error : error

  const centerTuple: [number, number] = roi 
    ? [ (roi.lat_min + roi.lat_max) / 2, (roi.lon_min + roi.lon_max) / 2 ]
    : [0, 180];

  if (displayLoading) {
    return <div className="map-loading">Loading Map...</div>;
  }

  if (displayError) {
    return <div className="map-error">Error: {displayError}</div>;
  }

  return (
    <MapContainer center={centerTuple} zoom={6} style={{ height: '100%', width: '100%' }}>
      {displayImageUrl && displayBounds && (
        <>
          <ImageOverlay url={displayImageUrl} bounds={displayBounds} />
          <FitBounds bounds={displayBounds} />
        </>
      )}

      {sitesGeoJson && (
        <GeoJSON 
          data={sitesGeoJson} 
          pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
          onEachFeature={(f, l) => onEachFeature(f, l, onSiteSelect)}
        />
      )}

      {waypointsGeoJson && (
        <GeoJSON 
          data={waypointsGeoJson} 
          pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
          style={(feature) => ({ color: feature?.properties?.line_color || '#ff0000', weight: 2, opacity: 0.8 })}
        />
      )}
    </MapContainer>
  );
}

