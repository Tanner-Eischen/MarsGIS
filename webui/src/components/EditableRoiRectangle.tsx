import { useEffect, useRef, useCallback } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';

const normalizeLon180 = (lon: number) => {
  let n = lon;
  while (n < -180) n += 360;
  while (n > 180) n -= 360;
  return n;
};

export interface RoiBounds {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

interface EditableRoiRectangleProps {
  roi: RoiBounds;
  onRoiChange: (roi: RoiBounds) => void;
  color?: string;
  interactive?: boolean;
}

const DEBOUNCE_MS = 300;
const HANDLE_RADIUS = 6;
const CENTER_HANDLE_RADIUS = 8;

export default function EditableRoiRectangle({
  roi,
  onRoiChange,
  color = '#06b6d4',
  interactive = true,
}: EditableRoiRectangleProps) {
  const map = useMap();
  const rectangleRef = useRef<L.Rectangle | null>(null);
  const markersRef = useRef<L.Marker[]>([]);
  const centerMarkerRef = useRef<L.Marker | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isDraggingRef = useRef(false);

  const roiToBounds = useCallback((r: RoiBounds) => {
    const south = Math.min(r.lat_min, r.lat_max);
    const north = Math.max(r.lat_min, r.lat_max);
    const west = normalizeLon180(r.lon_min);
    const east = normalizeLon180(r.lon_max);
    return L.latLngBounds(
      L.latLng(south, Math.min(west, east)),
      L.latLng(north, Math.max(west, east))
    );
  }, []);

  const boundsToRoi = useCallback((bounds: L.LatLngBounds): RoiBounds => {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    return {
      lat_min: Math.min(sw.lat, ne.lat),
      lat_max: Math.max(sw.lat, ne.lat),
      lon_min: normalizeLon180(Math.min(sw.lng, ne.lng)),
      lon_max: normalizeLon180(Math.max(sw.lng, ne.lng)),
    };
  }, []);

  const notifyRoi = useCallback(
    (newRoi: RoiBounds) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
      const fire = () => onRoiChange(newRoi);
      if (isDraggingRef.current) {
        debounceRef.current = setTimeout(fire, DEBOUNCE_MS);
      } else {
        fire();
      }
    },
    [onRoiChange]
  );

  useEffect(() => {
    const bounds = roiToBounds(roi);
    const layerGroup = L.layerGroup();

    const rect = L.rectangle(bounds, {
      color,
      weight: 2,
      dashArray: '6,4',
      fillColor: color,
      fillOpacity: 0.15,
    });
    rect.addTo(layerGroup);
    rectangleRef.current = rect;

    if (!interactive) {
      layerGroup.addTo(map);
      return () => {
        map.removeLayer(layerGroup);
        rectangleRef.current = null;
      };
    }

    const corners = [
      bounds.getSouthWest(),
      bounds.getSouthEast(),
      bounds.getNorthEast(),
      bounds.getNorthWest(),
    ];
    const oppositeIndex = (i: number) => (i + 2) % 4;

    const createHandleIcon = (isCenter: boolean) => {
      const r = isCenter ? CENTER_HANDLE_RADIUS : HANDLE_RADIUS;
      return L.divIcon({
        className: 'editable-roi-handle',
        html: `<div style="width:${r * 2}px;height:${r * 2}px;border-radius:50%;background:#fff;border:2px solid ${color};box-sizing:border-box;"></div>`,
        iconSize: [r * 2, r * 2],
        iconAnchor: [r, r],
      });
    };

    corners.forEach((latlng, i) => {
      const marker = L.marker(latlng, { icon: createHandleIcon(false), draggable: true });
      marker.addTo(layerGroup);
      markersRef.current[i] = marker;

      marker.on('dragstart', () => { isDraggingRef.current = true; });
      marker.on('dragend', () => {
        isDraggingRef.current = false;
        if (debounceRef.current) {
          clearTimeout(debounceRef.current);
          debounceRef.current = null;
        }
        const b = rect.getBounds();
        onRoiChange(boundsToRoi(b));
      });
      marker.on('drag', () => {
        const dragged = marker.getLatLng();
        const opposite = markersRef.current[oppositeIndex(i)]?.getLatLng();
        if (!opposite) return;
        const newBounds = L.latLngBounds(
          L.latLng(Math.min(dragged.lat, opposite.lat), Math.min(dragged.lng, opposite.lng)),
          L.latLng(Math.max(dragged.lat, opposite.lat), Math.max(dragged.lng, opposite.lng))
        );
        rect.setBounds(newBounds);
        const newCorners = [
          newBounds.getSouthWest(),
          newBounds.getSouthEast(),
          newBounds.getNorthEast(),
          newBounds.getNorthWest(),
        ];
        newCorners.forEach((ll, j) => markersRef.current[j]?.setLatLng(ll));
        centerMarkerRef.current?.setLatLng(newBounds.getCenter());
        notifyRoi(boundsToRoi(newBounds));
      });
    });

    const center = bounds.getCenter();
    const centerMarker = L.marker(center, { icon: createHandleIcon(true), draggable: true });
    centerMarker.addTo(layerGroup);
    centerMarkerRef.current = centerMarker;

    let startCenter: L.LatLng | null = null;
    let startCorners: L.LatLng[] = [];

    centerMarker.on('dragstart', () => {
      isDraggingRef.current = true;
      startCenter = centerMarker.getLatLng();
      startCorners = markersRef.current.map((m) => m.getLatLng().clone());
    });
    centerMarker.on('dragend', () => {
      isDraggingRef.current = false;
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
      const b = rect.getBounds();
      onRoiChange(boundsToRoi(b));
    });
    centerMarker.on('drag', () => {
      if (!startCenter) return;
      const current = centerMarker.getLatLng();
      const dLat = current.lat - startCenter.lat;
      const dLng = current.lng - startCenter.lng;
      const newCorners = startCorners.map((ll) => L.latLng(ll.lat + dLat, ll.lng + dLng));
      const newBounds = L.latLngBounds(newCorners);
      rect.setBounds(newBounds);
      newCorners.forEach((ll, j) => markersRef.current[j]?.setLatLng(ll));
      notifyRoi(boundsToRoi(newBounds));
    });

    layerGroup.addTo(map);

    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      map.removeLayer(layerGroup);
      rectangleRef.current = null;
      markersRef.current = [];
      centerMarkerRef.current = null;
    };
  }, [map, roi.lat_min, roi.lat_max, roi.lon_min, roi.lon_max, interactive, color, roiToBounds, boundsToRoi, notifyRoi, onRoiChange]);

  return null;
}
