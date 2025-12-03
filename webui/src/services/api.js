import axios from 'axios';
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api/v1';
const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000, // 30 second timeout for better user experience
});
// Add request interceptor for logging
api.interceptors.request.use((config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data ? { data: config.data } : '');
    return config;
}, (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
});
// Add response interceptor for logging
api.interceptors.response.use((response) => {
    console.log(`[API] Response ${response.status} from ${response.config.url}`);
    return response;
}, (error) => {
    console.error(`[API] Error ${error.response?.status || 'NETWORK'} from ${error.config?.url || 'unknown'}:`, error.message);
    if (error.response) {
        console.error('[API] Error details:', error.response.data);
    }
    return Promise.reject(error);
});
export const getStatus = async () => {
    const response = await api.get('/status');
    return response.data;
};
export const downloadDEM = async (request) => {
    const response = await api.post('/download', request);
    return response.data;
};
export const analyzeTerrain = async (request) => {
    const response = await api.post('/analyze', {
        dataset: request.dataset || 'mola',
        threshold: request.threshold || 0.7,
        roi: request.roi,
        criteria_weights: request.criteria_weights,
    });
    return response.data;
};
export const planNavigation = async (request) => {
    const response = await api.post('/navigation/plan-route', {
        site_id: request.site_id,
        analysis_dir: request.analysis_dir || 'data/output',
        start_lat: request.start_lat,
        start_lon: request.start_lon,
        max_waypoint_spacing_m: request.max_waypoint_spacing_m || 100.0,
        max_slope_deg: request.max_slope_deg || 25.0,
    });
    return response.data;
};
export const planMultipleRoutes = async (request) => {
    const response = await api.post('/navigation/plan-routes', {
        site_id: request.site_id,
        analysis_dir: request.analysis_dir,
        start_lat: request.start_lat,
        start_lon: request.start_lon,
        strategies: request.strategies || ['balanced', 'direct'],
        max_waypoint_spacing_m: request.max_waypoint_spacing_m || 100.0,
        max_slope_deg: request.max_slope_deg || 25.0,
        task_id: request.task_id,
    });
    return response.data;
};
export const runLandingScenario = async (request) => {
    const response = await api.post('/mission/landing-scenario', request);
    return response.data;
};
export const runTraverseScenario = async (request) => {
    const response = await api.post('/mission/rover-traverse', request);
    return response.data;
};
export const analyzeSolarPotential = async (request) => {
    const response = await api.post('/solar/analyze', request);
    return response.data;
};
export const getDemoMode = async () => {
    const response = await api.get('/demo-mode');
    return response.data;
};
export const setDemoMode = async (enabled) => {
    const response = await api.post('/demo-mode', { enabled });
    return response.data;
};
export const getExampleROIs = async () => {
    const response = await api.get('/examples/rois');
    return response.data;
};
