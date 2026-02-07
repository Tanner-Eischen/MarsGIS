/**
 * Terrain elevation sampling utilities for rover animation
 */
/**
 * Sample terrain elevation at a given lat/lon using bilinear interpolation
 * @param lat Latitude in degrees
 * @param lon Longitude in degrees
 * @param terrainGrid Terrain grid data with x, y, z arrays
 * @returns Interpolated elevation value
 */
export function sampleTerrainElevation(lat, lon, terrainGrid) {
    const { x, y, z } = terrainGrid;
    if (!x || !y || !z || x.length === 0 || y.length === 0) {
        return 0;
    }
    // Handle 2D coordinate arrays (each row may have different values)
    // Find the grid cell containing the point
    let minDist = Infinity;
    let bestRow = 0;
    let bestCol = 0;
    // Search for nearest grid point
    for (let i = 0; i < y.length; i++) {
        const row = y[i];
        if (!row || row.length === 0)
            continue;
        for (let j = 0; j < row.length; j++) {
            const gridLat = row[j];
            const gridLon = x[i]?.[j] ?? x[0]?.[j] ?? 0;
            const dist = Math.sqrt(Math.pow(gridLat - lat, 2) + Math.pow(gridLon - lon, 2));
            if (dist < minDist) {
                minDist = dist;
                bestRow = i;
                bestCol = j;
            }
        }
    }
    // Get surrounding grid points for bilinear interpolation
    const row = bestRow;
    const col = bestCol;
    // Get the four corner points
    const getValue = (r, c) => {
        if (r < 0 || r >= z.length)
            return 0;
        const zRow = z[r];
        if (!zRow || c < 0 || c >= zRow.length)
            return 0;
        return zRow[c] || 0;
    };
    const z11 = getValue(row, col);
    const z12 = getValue(row, col + 1);
    const z21 = getValue(row + 1, col);
    const z22 = getValue(row + 1, col + 1);
    // Get coordinate bounds
    const lat1 = y[row]?.[col] ?? lat;
    const lat2 = y[row + 1]?.[col] ?? lat;
    const lon1 = x[row]?.[col] ?? lon;
    const lon2 = x[row]?.[col + 1] ?? lon;
    // Simple interpolation if we're at the exact grid point
    if (Math.abs(lat - lat1) < 0.0001 && Math.abs(lon - lon1) < 0.0001) {
        return z11;
    }
    // Bilinear interpolation
    const dx = Math.abs(lon2 - lon1) > 0.0001
        ? (lon - lon1) / (lon2 - lon1)
        : 0;
    const dy = Math.abs(lat2 - lat1) > 0.0001
        ? (lat - lat1) / (lat2 - lat1)
        : 0;
    const clampedDx = Math.max(0, Math.min(1, dx));
    const clampedDy = Math.max(0, Math.min(1, dy));
    const elevation = z11 * (1 - clampedDx) * (1 - clampedDy) +
        z21 * clampedDx * (1 - clampedDy) +
        z12 * (1 - clampedDx) * clampedDy +
        z22 * clampedDx * clampedDy;
    return elevation;
}
/**
 * Calculate bearing (heading) between two points in degrees
 * @param lat1 Start latitude
 * @param lon1 Start longitude
 * @param lat2 End latitude
 * @param lon2 End longitude
 * @returns Bearing in degrees (0-360)
 */
export function calculateBearing(lat1, lon1, lat2, lon2) {
    const dLon = ((lon2 - lon1) * Math.PI) / 180;
    const lat1Rad = (lat1 * Math.PI) / 180;
    const lat2Rad = (lat2 * Math.PI) / 180;
    const y = Math.sin(dLon) * Math.cos(lat2Rad);
    const x = Math.cos(lat1Rad) * Math.sin(lat2Rad) -
        Math.sin(lat1Rad) * Math.cos(lat2Rad) * Math.cos(dLon);
    let bearing = (Math.atan2(y, x) * 180) / Math.PI;
    bearing = (bearing + 360) % 360; // Normalize to 0-360
    return bearing;
}
