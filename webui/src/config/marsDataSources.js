/**
 * Mars Geospatial Data Sources Configuration
 *
 * Defines overlay types, their metadata, and data source mappings for Mars datasets.
 * Supports both static (spatial-only) and temporal (spatiotemporal) overlays.
 */
/**
 * Mars overlay definitions with metadata
 */
export const MARS_OVERLAY_DEFINITIONS = [
    {
        name: 'elevation',
        displayName: 'Elevation',
        icon: '📊',
        description: 'Digital elevation model from MOLA/HiRISE/CTX',
        color: '#228B22',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx', 'mola_200m'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster',
        resolution: '463m (MOLA), 1m (HiRISE), 18m (CTX)'
    },
    {
        name: 'solar',
        displayName: 'Solar Potential',
        icon: '☀️',
        description: 'Solar irradiance potential with dust degradation',
        color: '#FFD700',
        temporalType: 'temporal',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster',
        temporalParams: {
            sunAzimuth: true,
            sunAltitude: true,
            marsSol: true,
            timeOfDay: true
        }
    },
    {
        name: 'viewshed',
        displayName: 'Comms / Viewshed',
        icon: '📡',
        description: 'Line-of-sight visibility from observer',
        color: '#00FF00',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster',
        resolution: 'Variable'
    },
    {
        name: 'comms_risk',
        displayName: 'Comms Risk Map',
        icon: '📶',
        description: 'Signal occlusion risk probability',
        color: '#FF4500',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster'
    },
    {
        name: 'dust',
        displayName: 'Dust Cover Index',
        icon: '🌪️',
        description: 'TES Dust Cover Index - surface dust coverage',
        color: '#8B7355',
        temporalType: 'temporal',
        datasets: ['mola'],
        backendEndpoint: '/api/v1/visualization/overlay',
        externalUrl: 'https://tes.asu.edu/~ruff/DCI/dci.html',
        dataType: 'raster',
        resolution: '~3.5 km per pixel',
        temporalParams: {
            marsSol: true,
            season: true,
            dustStormPeriod: true
        }
    },
    {
        name: 'hillshade',
        displayName: 'Hillshade',
        icon: '⛰️',
        description: '3D terrain shading',
        color: '#696969',
        temporalType: 'temporal',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster',
        temporalParams: {
            sunAzimuth: true,
            sunAltitude: true
        }
    },
    {
        name: 'slope',
        displayName: 'Slope',
        icon: '📐',
        description: 'Terrain slope in degrees',
        color: '#FF6347',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster'
    },
    {
        name: 'aspect',
        displayName: 'Aspect',
        icon: '🧭',
        description: 'Slope direction/orientation',
        color: '#4169E1',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster'
    },
    {
        name: 'roughness',
        displayName: 'Roughness',
        icon: '⛰️',
        description: 'Terrain roughness index',
        color: '#A0522D',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster'
    },
    {
        name: 'tri',
        displayName: 'TRI',
        icon: '🏔️',
        description: 'Terrain Ruggedness Index',
        color: '#8B4513',
        temporalType: 'static',
        datasets: ['mola', 'hirise', 'ctx'],
        backendEndpoint: '/api/v1/visualization/overlay',
        dataType: 'raster'
    },
    {
        name: 'nomenclature',
        displayName: 'Nomenclature',
        icon: '📍',
        description: 'Mars feature names and locations',
        color: '#32CD32',
        temporalType: 'static',
        datasets: ['mola'],
        backendEndpoint: '/api/v1/visualization/overlay',
        externalUrl: 'https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_nomenclature_polygons',
        dataType: 'vector'
    },
    {
        name: 'contours',
        displayName: 'Topography Contours',
        icon: '🗺️',
        description: 'Elevation contour lines',
        color: '#1E90FF',
        temporalType: 'static',
        datasets: ['mola'],
        backendEndpoint: '/api/v1/visualization/overlay',
        externalUrl: 'https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_contours_200m_lines',
        dataType: 'vector'
    },
    {
        name: 'albedo',
        displayName: 'Albedo',
        icon: '🌑',
        description: 'TES Albedo - surface reflectivity',
        color: '#C0C0C0',
        temporalType: 'static',
        datasets: ['mola'],
        backendEndpoint: '/api/v1/visualization/overlay',
        externalUrl: 'https://openplanetary.carto.com/u/opmbuilder/dataset/opm_499_mars_albedo_tes_7classes',
        dataType: 'vector'
    }
];
/**
 * Get overlay definition by name
 */
export function getOverlayDefinition(name) {
    return MARS_OVERLAY_DEFINITIONS.find(def => def.name === name);
}
/**
 * Get all overlays for a specific dataset
 */
export function getOverlaysForDataset(dataset) {
    return MARS_OVERLAY_DEFINITIONS.filter(def => def.datasets.includes(dataset));
}
/**
 * Check if overlay requires temporal parameters
 */
export function isTemporalOverlay(overlayType) {
    const def = getOverlayDefinition(overlayType);
    return def?.temporalType === 'temporal' || false;
}
/**
 * Get required temporal parameters for an overlay
 */
export function getTemporalParams(overlayType) {
    const def = getOverlayDefinition(overlayType);
    if (!def?.temporalParams)
        return [];
    const params = [];
    if (def.temporalParams.marsSol)
        params.push('marsSol');
    if (def.temporalParams.season)
        params.push('season');
    if (def.temporalParams.timeOfDay)
        params.push('timeOfDay');
    if (def.temporalParams.dustStormPeriod)
        params.push('dustStormPeriod');
    if (def.temporalParams.sunAzimuth)
        params.push('sunAzimuth');
    if (def.temporalParams.sunAltitude)
        params.push('sunAltitude');
    return params;
}
