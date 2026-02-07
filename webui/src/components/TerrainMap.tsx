import { MapContainer, ImageOverlay, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useSitesGeoJson, useWaypointsGeoJson, useOverlayImage } from '../hooks/useMapData';
import { useEffect, useRef } from 'react';

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
    if (bounds && bounds.isValid()) {
      console.log("Fitting bounds:", bounds);
      try {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 14, animate: true });
      } catch (e) {
        console.error("Error fitting bounds:", e);
      }
    }
  }, [map, bounds]);
  
  return null;
}

function MapController({ roi }) {
  const map = useMap();
  
  useEffect(() => {
    if (roi) {
        // Initial center if no bounds available yet
        const lat = (roi.lat_min + roi.lat_max) / 2;
        const lon = (roi.lon_min + roi.lon_max) / 2;
        map.panTo([lat, lon]);
    }
  }, [map, roi]);
  
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

// --- Layer Components ---

function SitesLayer({ showSites, onSiteSelect, selectedSiteId }) {
  const sitesGeoJson = useSitesGeoJson(showSites);
  if (!sitesGeoJson) return null;
  
  return (
    <GeoJSON 
      data={sitesGeoJson} 
      pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
      onEachFeature={(f, l) => onEachFeature(f, l, onSiteSelect)}
    />
  );
}

function WaypointsLayer({ showWaypoints, selectedSiteId }) {
  const waypointsGeoJson = useWaypointsGeoJson(showWaypoints);
  if (!waypointsGeoJson) return null;

  return (
    <GeoJSON 
      data={waypointsGeoJson} 
      pointToLayer={(f, l) => pointToLayer(f, l, selectedSiteId)}
      style={(feature) => ({ color: feature?.properties?.line_color || '#ff0000', weight: 2, opacity: 0.8 })}
    />
  );
}

// --- Main Component ---

interface TerrainMapProps {
  roi: any
  dataset?: string
  showSites?: boolean
  showWaypoints?: boolean
  relief?: number
  onSiteSelect?: any
  selectedSiteId?: any
  overlayType?:
    | 'elevation'
    | 'solar'
    | 'dust'
    | 'hillshade'
    | 'slope'
    | 'aspect'
    | 'roughness'
    | 'tri'
    | 'viewshed'
    | 'comms_risk'
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
  // Use overlay system for everything (including elevation) to benefit from caching
  const activeOverlayType = overlayType || 'elevation';
  
  const overlayImage = useOverlayImage(roi, dataset, activeOverlayType, {
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
  });

  const displayImageUrl = overlayImage?.url;
  const displayBounds = overlayImage?.bounds;
  const displayLoading = overlayImage?.loading;
  const displayError = overlayImage?.error;

  // Calculate center for initial map render
  const centerTuple: [number, number] = roi 
    ? [ (roi.lat_min + roi.lat_max) / 2, (roi.lon_min + roi.lon_max) / 2 ]
    : [0, 180];

  if (displayLoading) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-gray-900 text-cyan-400 font-mono">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400 mx-auto mb-2"></div>
          LOADING MAP DATA...
        </div>
      </div>
    );
  }

  if (displayError) {
    return (
      <div className="flex items-center justify-center h-full w-full bg-gray-900 text-red-400 font-mono border border-red-900/50">
        <div className="text-center p-4">
          <div className="text-xl mb-2">⚠ ERROR</div>
          <div>{displayError}</div>
        </div>
      </div>
    );
  }

  return (
    <MapContainer center={centerTuple} zoom={6} style={{ height: '100%', width: '100%' }} className="bg-black">
      <MapController roi={roi} />
      
      {displayImageUrl && displayBounds && (
        <>
          <ImageOverlay url={displayImageUrl} bounds={displayBounds} />
          <FitBounds bounds={displayBounds} />
        </>
      )}

      {showSites && (
        <SitesLayer showSites={showSites} onSiteSelect={onSiteSelect} selectedSiteId={selectedSiteId} />
      )}

      {showWaypoints && (
        <WaypointsLayer showWaypoints={showWaypoints} selectedSiteId={selectedSiteId} />
      )}
    </MapContainer>
  );
}
