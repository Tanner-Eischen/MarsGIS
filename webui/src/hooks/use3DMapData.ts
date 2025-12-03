import { useState, useEffect } from 'react';

export function use3DTerrain(roi, dataset, maxPoints = 50000) {
  const [terrainData, setTerrainData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!roi) return;

    const fetchTerrain = async () => {
      setLoading(true);
      setError(null);
      try {
        const roiStr = `${roi.lat_min},${roi.lat_max},${roi.lon_min},${roi.lon_max}`;
        const url = `http://localhost:5000/api/v1/visualization/terrain-3d?dataset=${dataset}&roi=${roiStr}&max_points=${maxPoints}`;
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to fetch 3D terrain data: ${response.statusText}`);
        }
        const data = await response.json();
        setTerrainData(data);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTerrain();
  }, [roi, dataset, maxPoints]);

  return { terrainData, loading, error };
}