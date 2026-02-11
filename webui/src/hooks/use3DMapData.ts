import { useState, useEffect } from 'react';
import { apiUrl } from '../lib/apiBase';

const buildApiError = (status: number, payload: any) => ({
  isServiceUnavailable: status === 503,
  status,
  message: payload?.detail || payload?.error || `Request failed (${status})`,
  detail: payload?.detail ?? null,
});

export function use3DTerrain(roi, dataset, maxPoints = 50000) {
  const [terrainData, setTerrainData] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!roi) return;

    const fetchTerrain = async () => {
      setLoading(true);
      setError(null);
      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`;
        const url = apiUrl(`/visualization/terrain-3d?dataset=${dataset}&roi=${roiStr}&max_points=${maxPoints}`);
        const response = await fetch(url);
        const data = await response.json();
        if (!response.ok) {
          throw buildApiError(response.status, data);
        }
        setTerrainData(data);
        setMetadata({
          datasetRequested: data.dataset_requested,
          datasetUsed: data.dataset_used,
          isFallback: data.is_fallback,
          fallbackReason: data.fallback_reason,
        });
      } catch (e: any) {
        if (e?.status) {
          setError(e);
        } else {
          setError({
            isServiceUnavailable: false,
            status: null,
            message: e?.message || 'Unknown error',
            detail: null,
          });
        }
      } finally {
        setLoading(false);
      }
    };

    fetchTerrain();
  }, [roi, dataset, maxPoints]);

  return { terrainData, metadata, loading, error };
}
