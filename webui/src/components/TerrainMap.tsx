import { MapContainer, ImageOverlay, GeoJSON, TileLayer, useMap } from 'react-leaflet';
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

const MARS_BASEMAP_URL =
  'https://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/celestia_mars-shaded-16k_global/{z}/{x}/{y}.png';

function FitBounds({ bounds, fitKey }) {
  const map = useMap();
  const lastFitKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (bounds && bounds.isValid() && lastFitKeyRef.current !== fitKey) {
      try {
        map.fitBounds(bounds, { padding: [20, 20], maxZoom: 14, animate: true });
        lastFitKeyRef.current = fitKey;
      } catch (e) {
        console.error('Error fitting bounds:', e);
      }
    }
  }, [map, bounds, fitKey]);

  return null;
}

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
        Area: ${props.area_km2.toFixed(2)} km2<br/>
        Suitability: ${(props.suitability_score * 100).toFixed(1)}%<br/>
        Slope: ${props.mean_slope_deg.toFixed(2)} deg<br/>
        Elevation: ${props.mean_elevation_m.toFixed(0)} m
      </div>
    `;
    layer.bindPopup(popupContent);
    if (onSiteSelect) {
      layer.on('click', () => onSiteSelect(props.site_id));
    }
  }
};

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
  const activeOverlayType = overlayType || 'elevation';

  const overlayImage = useOverlayImage(roi, dataset, activeOverlayType, {
    colormap: overlayOptions.colormap || 'terrain',
    relief: overlayOptions.relief ?? relief,
    sunAzimuth: overlayOptions.sunAzimuth || 315,
    sunAltitude: overlayOptions.sunAltitude || 45,
    width: overlayOptions.width || 1200,
    height: overlayOptions.height || 800,
    buffer: overlayOptions.buffer || 0.25,
    marsSol: overlayOptions.marsSol,
    season: overlayOptions.season,
    dustStormPeriod: overlayOptions.dustStormPeriod
  });

  const displayImageUrl = overlayImage?.url;
  const displayBounds = overlayImage?.bounds;
  const displayLoading = overlayImage?.loading;
  const displayError = overlayImage?.error;

  const centerTuple: [number, number] = roi
    ? [
        (roi.lat_min + roi.lat_max) / 2,
        (() => {
          const lon = (roi.lon_min + roi.lon_max) / 2;
          return lon > 180 ? lon - 360 : lon;
        })(),
      ]
    : [0, 180];

  const fitKey = roi
    ? `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}|${dataset}|${activeOverlayType}`
    : `${dataset}|${activeOverlayType}`;

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={centerTuple}
        zoom={6}
        style={{ height: '100%', width: '100%' }}
        className="bg-black"
        scrollWheelZoom={true}
      >
        <TileLayer
          url={MARS_BASEMAP_URL}
          attribution='&copy; USGS / OpenPlanetary'
          maxZoom={8}
          noWrap={true}
        />

        {displayImageUrl && displayBounds && (
          <>
            <ImageOverlay url={displayImageUrl} bounds={displayBounds} opacity={0.78} />
            <FitBounds bounds={displayBounds} fitKey={fitKey} />
          </>
        )}

        {showSites && (
          <SitesLayer showSites={showSites} onSiteSelect={onSiteSelect} selectedSiteId={selectedSiteId} />
        )}

        {showWaypoints && (
          <WaypointsLayer showWaypoints={showWaypoints} selectedSiteId={selectedSiteId} />
        )}
      </MapContainer>

      {displayLoading && (
        <div className="pointer-events-none absolute top-3 left-3 bg-gray-900/85 border border-cyan-700/60 text-cyan-300 text-xs font-mono px-3 py-2 rounded">
          Loading map data...
        </div>
      )}

      {displayError && (
        <div className="absolute top-3 right-3 max-w-[420px] bg-red-950/90 border border-red-700/70 text-red-200 text-xs font-mono px-3 py-2 rounded">
          Map data error: {displayError}
        </div>
      )}
    </div>
  );
}
