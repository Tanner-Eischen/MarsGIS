import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { Brain, TrendingUp, AlertTriangle, CheckCircle, BarChart3, Target } from 'lucide-react';
import { apiFetch } from '../lib/apiBase';
const MISSION_TYPES = [
    { value: 'research_base', label: 'Research Base', description: 'Scientific research facility' },
    { value: 'mining_operation', label: 'Mining Operation', description: 'Resource extraction site' },
    { value: 'emergency_shelter', label: 'Emergency Shelter', description: 'Emergency refuge station' },
    { value: 'permanent_settlement', label: 'Permanent Settlement', description: 'Long-term colony site' }
];
const SAMPLE_TRAINING_DATA = [
    {
        terrain: {
            elevation: -2000,
            slope_mean: 5.2,
            aspect_mean: 135.0,
            roughness_mean: 0.3,
            tri_mean: 15.0,
            coordinates: [18.0, 77.0],
            slope_std: 2.1,
            roughness_std: 0.1,
            elevation_std: 50.0
        },
        mission_success_score: 0.85,
        mission_type: "research_base",
        site_name: "Olympus Mons Base Alpha"
    },
    {
        terrain: {
            elevation: -4000,
            slope_mean: 12.5,
            aspect_mean: 45.0,
            roughness_mean: 0.8,
            tri_mean: 35.0,
            coordinates: [-15.0, 175.0],
            slope_std: 5.2,
            roughness_std: 0.3,
            elevation_std: 120.0
        },
        mission_success_score: 0.72,
        mission_type: "mining_operation",
        site_name: "Syrtis Major Mining Outpost"
    },
    {
        terrain: {
            elevation: 1000,
            slope_mean: 2.1,
            aspect_mean: 180.0,
            roughness_mean: 0.1,
            tri_mean: 8.0,
            coordinates: [32.0, 91.0],
            slope_std: 0.8,
            roughness_std: 0.05,
            elevation_std: 25.0
        },
        mission_success_score: 0.92,
        mission_type: "emergency_shelter",
        site_name: "Protonilus Emergency Station"
    },
    {
        terrain: {
            elevation: -1500,
            slope_mean: 8.7,
            aspect_mean: 90.0,
            roughness_mean: 0.5,
            tri_mean: 22.0,
            coordinates: [55.0, 150.0],
            slope_std: 3.1,
            roughness_std: 0.2,
            elevation_std: 80.0
        },
        mission_success_score: 0.78,
        mission_type: "permanent_settlement",
        site_name: "Arcadia Planitia Colony"
    },
    {
        terrain: {
            elevation: -3500,
            slope_mean: 15.2,
            aspect_mean: 270.0,
            roughness_mean: 1.2,
            tri_mean: 45.0,
            coordinates: [-8.0, 282.0],
            slope_std: 7.8,
            roughness_std: 0.6,
            elevation_std: 200.0
        },
        mission_success_score: 0.45,
        mission_type: "research_base",
        site_name: "Valles Marineris Research Site"
    },
    {
        terrain: {
            elevation: 3000,
            slope_mean: 1.8,
            aspect_mean: 200.0,
            roughness_mean: 0.2,
            tri_mean: 12.0,
            coordinates: [0.0, 110.0],
            slope_std: 1.2,
            roughness_std: 0.08,
            elevation_std: 40.0
        },
        mission_success_score: 0.88,
        mission_type: "mining_operation",
        site_name: "Arabia Terra Mining Complex"
    }
];
const SAMPLE_CANDIDATE_SITES = [
    {
        coordinates: [18.0, 77.0],
        elevation: -2000,
        slope_mean: 5.2,
        aspect_mean: 135.0,
        roughness_mean: 0.3,
        tri_mean: 15.0
    },
    {
        coordinates: [-15.0, 175.0],
        elevation: -4000,
        slope_mean: 12.5,
        aspect_mean: 45.0,
        roughness_mean: 0.8,
        tri_mean: 35.0
    },
    {
        coordinates: [32.0, 91.0],
        elevation: 1000,
        slope_mean: 2.1,
        aspect_mean: 180.0,
        roughness_mean: 0.1,
        tri_mean: 8.0
    },
    {
        coordinates: [55.0, 150.0],
        elevation: -1500,
        slope_mean: 8.7,
        aspect_mean: 90.0,
        roughness_mean: 0.5,
        tri_mean: 22.0
    },
    {
        coordinates: [-8.0, 282.0],
        elevation: -3500,
        slope_mean: 15.2,
        aspect_mean: 270.0,
        roughness_mean: 1.2,
        tri_mean: 45.0
    },
    {
        coordinates: [0.0, 110.0],
        elevation: 3000,
        slope_mean: 1.8,
        aspect_mean: 200.0,
        roughness_mean: 0.2,
        tri_mean: 12.0
    }
];
export default function MLSiteRecommendation({ candidateSites = SAMPLE_CANDIDATE_SITES, className = '', onRecommendationComplete, destinationCoordinates }) {
    const [recommendations, setRecommendations] = useState([]);
    const [modelInsights, setModelInsights] = useState(null);
    const [loading, setLoading] = useState(false);
    const [trainingLoading, setTrainingLoading] = useState(false);
    const [selectedMission, setSelectedMission] = useState('research_base');
    const [topN, setTopN] = useState(3);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('recommendations');
    const fetchModelInsights = async () => {
        try {
            const response = await apiFetch('/ml-recommendation/insights');
            if (!response.ok)
                throw new Error('Failed to fetch model insights');
            const data = await response.json();
            if (data.success) {
                setModelInsights(data);
            }
        }
        catch (err) {
            console.error('Error fetching model insights:', err);
        }
    };
    const trainModels = async () => {
        setTrainingLoading(true);
        setError(null);
        try {
            const response = await apiFetch('/ml-recommendation/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    historical_sites: [], // Use server-side sample data
                    mission_type: selectedMission
                })
            });
            if (!response.ok)
                throw new Error('Training failed');
            const data = await response.json();
            if (data.success) {
                await fetchModelInsights();
                // Auto-generate recommendations after training
                generateRecommendations();
            }
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Training failed');
        }
        finally {
            setTrainingLoading(false);
        }
    };
    const generateRecommendations = async () => {
        setLoading(true);
        setError(null);
        try {
            const endpoint = destinationCoordinates ? '/ml-recommendation/recommend-for-destination' : '/ml-recommendation/recommend';
            const payload = destinationCoordinates ? {
                destination_coordinates: destinationCoordinates,
                candidate_sites: candidateSites,
                mission_type: selectedMission,
                top_n: topN
            } : {
                candidate_sites: candidateSites,
                mission_type: selectedMission,
                top_n: topN
            };
            const response = await apiFetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok)
                throw new Error('Recommendation failed');
            const data = await response.json();
            if (data.success) {
                setRecommendations(data.recommendations);
                if (onRecommendationComplete) {
                    onRecommendationComplete(data.recommendations);
                }
            }
        }
        catch (err) {
            setError(err instanceof Error ? err.message : 'Recommendation failed');
        }
        finally {
            setLoading(false);
        }
    };
    useEffect(() => {
        if (activeTab === 'insights' && !modelInsights) {
            fetchModelInsights();
        }
    }, [activeTab]);
    const getScoreColor = (score) => {
        if (score >= 0.8)
            return 'text-green-600';
        if (score >= 0.6)
            return 'text-yellow-600';
        if (score >= 0.4)
            return 'text-orange-600';
        return 'text-red-600';
    };
    const getConfidenceColor = (confidence) => {
        if (confidence >= 0.8)
            return 'bg-green-100 text-green-800';
        if (confidence >= 0.6)
            return 'bg-yellow-100 text-yellow-800';
        return 'bg-red-100 text-red-800';
    };
    const getCategoryColor = (category) => {
        switch (category) {
            case 'Excellent': return 'bg-green-100 text-green-800';
            case 'Good': return 'bg-blue-100 text-blue-800';
            case 'Fair': return 'bg-yellow-100 text-yellow-800';
            case 'Poor': return 'bg-red-100 text-red-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    };
    const FeatureRadarChart = ({ scores }) => {
        const features = [
            { name: 'Safety', value: scores.safety, color: 'bg-red-500' },
            { name: 'Accessibility', value: scores.accessibility, color: 'bg-blue-500' },
            { name: 'Resources', value: scores.resources, color: 'bg-green-500' },
            { name: 'Stability', value: scores.stability, color: 'bg-purple-500' },
            { name: 'Solar', value: scores.solar_exposure, color: 'bg-yellow-500' },
            { name: 'Terrain', value: scores.terrain, color: 'bg-gray-500' }
        ];
        return (_jsx("div", { className: "grid grid-cols-2 gap-2", children: features.map((feature) => (_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: `w-3 h-3 rounded-full ${feature.color}` }), _jsxs("div", { className: "flex-1", children: [_jsxs("div", { className: "flex justify-between text-xs", children: [_jsx("span", { children: feature.name }), _jsxs("span", { className: getScoreColor(feature.value), children: [(feature.value * 100).toFixed(0), "%"] })] }), _jsx("div", { className: "w-full bg-gray-200 rounded-full h-1", children: _jsx("div", { className: `h-1 rounded-full ${feature.color}`, style: { width: `${feature.value * 100}%` } }) })] })] }, feature.name))) }));
    };
    return (_jsxs("div", { className: `bg-gray-800 rounded-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-6", children: [_jsxs("div", { className: "flex items-center space-x-3", children: [_jsx(Brain, { className: "w-8 h-8 text-purple-400" }), _jsxs("div", { children: [_jsx("h2", { className: "text-2xl font-bold text-gray-100", children: "ML Site Recommendation" }), _jsx("p", { className: "text-gray-300", children: "AI-powered habitat placement optimization" })] })] }), _jsxs("div", { className: "flex space-x-2", children: [_jsxs("button", { onClick: () => setActiveTab('recommendations'), className: `px-4 py-2 rounded-lg font-medium ${activeTab === 'recommendations'
                                    ? 'bg-purple-600 text-white'
                                    : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}`, children: [_jsx(Target, { className: "w-4 h-4 inline mr-2" }), "Recommendations"] }), _jsxs("button", { onClick: () => setActiveTab('insights'), className: `px-4 py-2 rounded-lg font-medium ${activeTab === 'insights'
                                    ? 'bg-purple-600 text-white'
                                    : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}`, children: [_jsx(BarChart3, { className: "w-4 h-4 inline mr-2" }), "Model Insights"] }), _jsxs("button", { onClick: () => setActiveTab('training'), className: `px-4 py-2 rounded-lg font-medium ${activeTab === 'training'
                                    ? 'bg-purple-600 text-white'
                                    : 'bg-gray-700 text-gray-200 hover:bg-gray-600'}`, children: [_jsx(TrendingUp, { className: "w-4 h-4 inline mr-2" }), "Training"] })] })] }), _jsx("div", { className: "bg-gray-700 rounded-lg p-4 mb-6", children: _jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-200 mb-2", children: "Mission Type" }), _jsx("select", { value: selectedMission, onChange: (e) => setSelectedMission(e.target.value), className: "w-full px-3 py-2 bg-gray-800 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500", children: MISSION_TYPES.map((type) => (_jsxs("option", { value: type.value, children: [type.label, " - ", type.description] }, type.value))) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-200 mb-2", children: "Top N Recommendations" }), _jsx("select", { value: topN, onChange: (e) => setTopN(Number(e.target.value)), className: "w-full px-3 py-2 bg-gray-800 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500", children: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (_jsx("option", { value: n, children: n }, n))) })] }), _jsx("div", { className: "flex items-end", children: _jsx("button", { onClick: generateRecommendations, disabled: loading, className: "w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed", children: loading ? 'Analyzing...' : 'Generate Recommendations' }) })] }) }), error && (_jsx("div", { className: "bg-red-900/30 border border-red-700 rounded-lg p-4 mb-6", children: _jsxs("div", { className: "flex items-center", children: [_jsx(AlertTriangle, { className: "w-5 h-5 text-red-400 mr-2" }), _jsx("span", { className: "text-red-200", children: error })] }) })), activeTab === 'recommendations' && (_jsx("div", { className: "space-y-4", children: recommendations.length === 0 ? (_jsxs("div", { className: "text-center py-12", children: [_jsx(Brain, { className: "w-16 h-16 text-gray-400 mx-auto mb-4" }), _jsx("p", { className: "text-gray-300", children: "No recommendations available. Generate recommendations to see ML-powered site analysis." })] })) : (recommendations.map((recommendation) => (_jsxs("div", { className: "border border-gray-700 rounded-lg p-6", children: [_jsx("div", { className: "flex items-start justify-between mb-4", children: _jsxs("div", { children: [_jsxs("div", { className: "flex items-center space-x-3 mb-2", children: [_jsx("div", { className: "flex items-center justify-center w-8 h-8 bg-purple-100 text-purple-800 rounded-full font-bold", children: recommendation.suitability_rank }), _jsxs("div", { children: [_jsxs("h3", { className: "text-lg font-semibold text-gray-100", children: ["Site ", recommendation.site_id] }), _jsxs("p", { className: "text-sm text-gray-300", children: [recommendation.coordinates[0].toFixed(2), "\u00B0, ", recommendation.coordinates[1].toFixed(2), "\u00B0"] })] })] }), _jsxs("div", { className: "flex items-center space-x-4", children: [_jsx("div", { className: `px-3 py-1 rounded-full text-sm font-medium ${getCategoryColor(recommendation.suitability_category)}`, children: recommendation.suitability_category }), _jsxs("div", { className: `px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(recommendation.confidence)}`, children: ["Confidence: ", (recommendation.confidence * 100).toFixed(0), "%"] }), _jsxs("div", { className: "flex items-center space-x-1", children: [_jsx("span", { className: "text-sm text-gray-300", children: "Score:" }), _jsxs("span", { className: `font-bold ${getScoreColor(recommendation.overall_score)}`, children: [(recommendation.overall_score * 100).toFixed(1), "%"] })] })] })] }) }), _jsxs("div", { className: "grid grid-cols-1 lg:grid-cols-2 gap-6", children: [_jsxs("div", { children: [_jsx("h4", { className: "font-medium text-gray-100 mb-3", children: "Feature Analysis" }), _jsx(FeatureRadarChart, { scores: recommendation.feature_scores })] }), _jsx("div", { children: _jsxs("div", { className: "space-y-4", children: [recommendation.recommendation_reasons.length > 0 && (_jsxs("div", { children: [_jsxs("h4", { className: "font-medium text-gray-100 mb-2 flex items-center", children: [_jsx(CheckCircle, { className: "w-4 h-4 text-green-600 mr-2" }), "Strengths"] }), _jsx("ul", { className: "space-y-1", children: recommendation.recommendation_reasons.map((reason, idx) => (_jsxs("li", { className: "text-sm text-gray-300 flex items-start", children: [_jsx("span", { className: "w-1.5 h-1.5 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0" }), reason] }, idx))) })] })), recommendation.risk_factors.length > 0 && (_jsxs("div", { children: [_jsxs("h4", { className: "font-medium text-gray-100 mb-2 flex items-center", children: [_jsx(AlertTriangle, { className: "w-4 h-4 text-orange-600 mr-2" }), "Risk Factors"] }), _jsx("ul", { className: "space-y-1", children: recommendation.risk_factors.map((risk, idx) => (_jsxs("li", { className: "text-sm text-gray-300 flex items-start", children: [_jsx("span", { className: "w-1.5 h-1.5 bg-orange-500 rounded-full mt-2 mr-2 flex-shrink-0" }), risk] }, idx))) })] }))] }) })] }), _jsx("div", { className: "mt-4 pt-4 border-t border-gray-200", children: _jsxs("div", { className: "flex items-center justify-between text-sm text-gray-300", children: [_jsxs("span", { children: ["Cluster Assignment: #", recommendation.cluster_assignment] }), _jsxs("span", { children: ["Analysis for: ", MISSION_TYPES.find(t => t.value === selectedMission)?.label] })] }) })] }, recommendation.site_id)))) })), activeTab === 'insights' && (_jsx("div", { className: "space-y-6", children: modelInsights ? (_jsxs(_Fragment, { children: [_jsxs("div", { className: "grid grid-cols-1 md:grid-cols-3 gap-4", children: [_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-2", children: "Model Status" }), _jsx("p", { className: "text-gray-300", children: modelInsights && 'model_status' in modelInsights && modelInsights.model_status === 'trained' ? '✅ Trained & Ready' : '❌ Not Trained' })] }), modelInsights.performance_metrics?.r2_score && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-2", children: "R\u00B2 Score" }), _jsxs("p", { className: "text-2xl font-bold text-gray-200", children: [(modelInsights.performance_metrics.r2_score * 100).toFixed(1), "%"] })] })), modelInsights.clustering_info?.n_clusters && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-2", children: "Clusters" }), _jsx("p", { className: "text-2xl font-bold text-gray-200", children: modelInsights.clustering_info.n_clusters })] }))] }), Object.keys(modelInsights.feature_importance).length > 0 && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-4", children: "Feature Importance" }), _jsx("div", { className: "space-y-3", children: Object.entries(modelInsights.feature_importance).map(([feature, importance]) => (_jsxs("div", { className: "flex items-center justify-between", children: [_jsx("span", { className: "text-sm font-medium text-gray-200 capitalize", children: feature.replace('_', ' ') }), _jsx("span", { className: "text-sm font-bold text-gray-100", children: importance })] }, feature))) })] })), modelInsights.training_info?.samples && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-2", children: "Training Information" }), _jsxs("div", { className: "space-y-2 text-sm text-gray-300", children: [_jsxs("p", { children: [_jsx("strong", { children: "Samples:" }), " ", modelInsights.training_info.samples] }), modelInsights.training_info.training_date && (_jsxs("p", { children: [_jsx("strong", { children: "Last Trained:" }), " ", new Date(modelInsights.training_info.training_date).toLocaleDateString()] }))] })] }))] })) : (_jsxs("div", { className: "text-center py-12", children: [_jsx(BarChart3, { className: "w-16 h-16 text-gray-400 mx-auto mb-4" }), _jsx("p", { className: "text-gray-300", children: "No model insights available. Train the models first." })] })) })), activeTab === 'training' && (_jsx("div", { className: "space-y-6", children: _jsxs("div", { className: "bg-gray-700 rounded-lg p-6", children: [_jsx("h3", { className: "font-semibold text-gray-100 mb-4", children: "Model Training" }), _jsx("p", { className: "text-gray-300 mb-4", children: "Train the machine learning models using historical Mars mission data to improve recommendation accuracy." }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsx("h4", { className: "font-medium text-gray-100 mb-2", children: "Training Data" }), _jsxs("p", { className: "text-sm text-gray-300", children: ["Models will be trained on ", SAMPLE_TRAINING_DATA.length, " historical Mars mission sites with known outcomes."] })] }), _jsxs("div", { children: [_jsx("h4", { className: "font-medium text-gray-100 mb-2", children: "Model Features" }), _jsxs("div", { className: "grid grid-cols-2 gap-2 text-sm text-gray-300", children: [_jsx("div", { children: "\u2022 Elevation" }), _jsx("div", { children: "\u2022 Slope" }), _jsx("div", { children: "\u2022 Aspect" }), _jsx("div", { children: "\u2022 Roughness" }), _jsx("div", { children: "\u2022 TRI" }), _jsx("div", { children: "\u2022 Solar Exposure" }), _jsx("div", { children: "\u2022 Resource Proximity" }), _jsx("div", { children: "\u2022 Safety Score" })] })] })] }), _jsx("button", { onClick: trainModels, disabled: trainingLoading, className: "mt-4 px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed", children: trainingLoading ? 'Training Models...' : 'Train ML Models' })] }) }))] }));
}
