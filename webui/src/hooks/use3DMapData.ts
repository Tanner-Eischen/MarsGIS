import { useState, useEffect } from 'react';
import { apiUrl } from '../lib/apiBase';

interface Roi {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
}

interface TerrainMetadata {
  datasetRequested: string;
  datasetUsed: string;
  isFallback: boolean;
  fallbackReason: string | null;
}

interface TerrainError {
  message: string;
  status: number | null;
  detail: string | null;
  isServiceUnavailable: boolean;
}

const parseErrorDetail = async (response: Response): Promise<string | null> => {
  try {
    const payload = await response.clone().json();
    if (typeof payload === 'string') {
      return payload;
    }
    if (typeof payload?.detail === 'string') {
      return payload.detail;
    }
    if (payload?.detail !== undefined) {
      return JSON.stringify(payload.detail);
    }
    if (typeof payload?.message === 'string') {
      return payload.message;
    }
    if (payload != null) {
      return JSON.stringify(payload);
    }
  } catch {
    // Ignore JSON parse failures and fall back to text payload.
  }

  try {
    const text = await response.text();
    const trimmed = text.trim();
    return trimmed.length > 0 ? trimmed : null;
  } catch {
    return null;
  }
};

export function use3DTerrain(roi: Roi | null, dataset: string, maxPoints = 50000) {
  const [terrainData, setTerrainData] = useState<any>(null);
  const [metadata, setMetadata] = useState<TerrainMetadata | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<TerrainError | null>(null);

  useEffect(() => {
    if (!roi) {
      setTerrainData(null);
      setMetadata(null);
      setError(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    const fetchTerrain = async () => {
      setLoading(true);
      setError(null);
      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`;
        const url = apiUrl(`/visualization/terrain-3d?dataset=${dataset}&roi=${roiStr}&max_points=${maxPoints}`);
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok) {
          const detail = await parseErrorDetail(response);
          const isServiceUnavailable = response.status === 503;
          const message = isServiceUnavailable
            ? '3D terrain service is temporarily unavailable (503).'
            : `Failed to fetch 3D terrain data (${response.status} ${response.statusText || 'error'})`;
          setTerrainData(null);
          setMetadata(null);
          setError({
            message,
            status: response.status,
            detail,
            isServiceUnavailable,
          });
          return;
        }
        const data = await response.json();
        if (cancelled) return;
        setTerrainData(data);
        setMetadata({
          datasetRequested: data.dataset_requested || dataset,
          datasetUsed: data.dataset_used || dataset,
          isFallback: Boolean(data.is_fallback),
          fallbackReason: data.fallback_reason || null,
        });
      } catch (e) {
        if (cancelled) return;
        if (e instanceof Error && e.name === 'AbortError') {
          return;
        }
        setTerrainData(null);
        setMetadata(null);
        const message = e instanceof Error ? e.message : 'Unknown 3D terrain loading error';
        setError({
          message,
          status: null,
          detail: null,
          isServiceUnavailable: false,
        });
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchTerrain();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [roi?.lat_min, roi?.lat_max, roi?.lon_min, roi?.lon_max, dataset, maxPoints]);

  return { terrainData, metadata, loading, error };
}
