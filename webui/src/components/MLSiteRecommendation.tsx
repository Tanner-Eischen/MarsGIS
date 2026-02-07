import { useState, useEffect } from 'react';
import { Brain, TrendingUp, AlertTriangle, CheckCircle, BarChart3, Target } from 'lucide-react';
import { apiFetch } from '../lib/apiBase';

interface SiteFeatures {
  safety: number;
  accessibility: number;
  resources: number;
  stability: number;
  solar_exposure: number;
  terrain: number;
}

interface SiteRecommendation {
  site_id: string;
  coordinates: [number, number];
  overall_score: number;
  feature_scores: SiteFeatures;
  confidence: number;
  recommendation_reasons: string[];
  risk_factors: string[];
  suitability_rank: number;
  cluster_assignment: number;
  suitability_category: string;
}

interface MLRecommendationRequest {
  candidate_sites: Array<{
    coordinates: [number, number];
    elevation: number;
    slope_mean: number;
    aspect_mean: number;
    roughness_mean: number;
    tri_mean: number;
    slope_std?: number;
    roughness_std?: number;
    elevation_std?: number;
  }>;
  mission_type: string;
  top_n: number;
}

interface ModelInsights {
  performance_metrics: {
    r2_score?: number;
    rmse?: number;
    mae?: number;
    silhouette_score?: number;
  };
  feature_importance: Record<string, string>;
  clustering_info: {
    n_clusters?: number;
    silhouette_score?: number;
  };
  training_info: {
    samples?: number;
    training_date?: string;
  };
}

interface MLSiteRecommendationProps {
  candidateSites?: Array<{
    coordinates: [number, number];
    elevation: number;
    slope_mean: number;
    aspect_mean: number;
    roughness_mean: number;
    tri_mean: number;
  }>;
  className?: string;
  onRecommendationComplete?: (recommendations: SiteRecommendation[]) => void;
  destinationCoordinates?: [number, number];
}

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
    coordinates: [18.0, 77.0] as [number, number],
    elevation: -2000,
    slope_mean: 5.2,
    aspect_mean: 135.0,
    roughness_mean: 0.3,
    tri_mean: 15.0
  },
  {
    coordinates: [-15.0, 175.0] as [number, number],
    elevation: -4000,
    slope_mean: 12.5,
    aspect_mean: 45.0,
    roughness_mean: 0.8,
    tri_mean: 35.0
  },
  {
    coordinates: [32.0, 91.0] as [number, number],
    elevation: 1000,
    slope_mean: 2.1,
    aspect_mean: 180.0,
    roughness_mean: 0.1,
    tri_mean: 8.0
  },
  {
    coordinates: [55.0, 150.0] as [number, number],
    elevation: -1500,
    slope_mean: 8.7,
    aspect_mean: 90.0,
    roughness_mean: 0.5,
    tri_mean: 22.0
  },
  {
    coordinates: [-8.0, 282.0] as [number, number],
    elevation: -3500,
    slope_mean: 15.2,
    aspect_mean: 270.0,
    roughness_mean: 1.2,
    tri_mean: 45.0
  },
  {
    coordinates: [0.0, 110.0] as [number, number],
    elevation: 3000,
    slope_mean: 1.8,
    aspect_mean: 200.0,
    roughness_mean: 0.2,
    tri_mean: 12.0
  }
];

export default function MLSiteRecommendation({ 
  candidateSites = SAMPLE_CANDIDATE_SITES,
  className = '',
  onRecommendationComplete,
  destinationCoordinates
}: MLSiteRecommendationProps) {
  const [recommendations, setRecommendations] = useState<SiteRecommendation[]>([]);
  const [modelInsights, setModelInsights] = useState<ModelInsights | null>(null);
  const [loading, setLoading] = useState(false);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [selectedMission, setSelectedMission] = useState('research_base');
  const [topN, setTopN] = useState(3);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'recommendations' | 'insights' | 'training'>('recommendations');

  const fetchModelInsights = async () => {
    try {
      const response = await apiFetch('/ml-recommendation/insights');
      if (!response.ok) throw new Error('Failed to fetch model insights');
      
      const data = await response.json();
      if (data.success) {
        setModelInsights(data);
      }
    } catch (err) {
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
      
      if (!response.ok) throw new Error('Training failed');
      
      const data = await response.json();
      if (data.success) {
        await fetchModelInsights();
        // Auto-generate recommendations after training
        generateRecommendations();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
    } finally {
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
      
      if (!response.ok) throw new Error('Recommendation failed');
      
      const data = await response.json();
      if (data.success) {
        setRecommendations(data.recommendations);
        if (onRecommendationComplete) {
          onRecommendationComplete(data.recommendations);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Recommendation failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'insights' && !modelInsights) {
      fetchModelInsights();
    }
  }, [activeTab]);

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    if (score >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'Excellent': return 'bg-green-100 text-green-800';
      case 'Good': return 'bg-blue-100 text-blue-800';
      case 'Fair': return 'bg-yellow-100 text-yellow-800';
      case 'Poor': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const FeatureRadarChart = ({ scores }: { scores: SiteFeatures }) => {
    const features = [
      { name: 'Safety', value: scores.safety, color: 'bg-red-500' },
      { name: 'Accessibility', value: scores.accessibility, color: 'bg-blue-500' },
      { name: 'Resources', value: scores.resources, color: 'bg-green-500' },
      { name: 'Stability', value: scores.stability, color: 'bg-purple-500' },
      { name: 'Solar', value: scores.solar_exposure, color: 'bg-yellow-500' },
      { name: 'Terrain', value: scores.terrain, color: 'bg-gray-500' }
    ];

    return (
      <div className="grid grid-cols-2 gap-2">
        {features.map((feature) => (
          <div key={feature.name} className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${feature.color}`}></div>
            <div className="flex-1">
              <div className="flex justify-between text-xs">
                <span>{feature.name}</span>
                <span className={getScoreColor(feature.value)}>
                  {(feature.value * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1">
                <div 
                  className={`h-1 rounded-full ${feature.color}`}
                  style={{ width: `${feature.value * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Brain className="w-8 h-8 text-purple-400" />
          <div>
            <h2 className="text-2xl font-bold text-gray-100">ML Site Recommendation</h2>
            <p className="text-gray-300">AI-powered habitat placement optimization</p>
          </div>
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('recommendations')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'recommendations' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
            }`}
          >
            <Target className="w-4 h-4 inline mr-2" />
            Recommendations
          </button>
          <button
            onClick={() => setActiveTab('insights')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'insights' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-2" />
            Model Insights
          </button>
          <button
            onClick={() => setActiveTab('training')}
            className={`px-4 py-2 rounded-lg font-medium ${
              activeTab === 'training' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-700 text-gray-200 hover:bg-gray-600'
            }`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            Training
          </button>
        </div>
      </div>

      <div className="bg-gray-700 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Mission Type
            </label>
            <select
              value={selectedMission}
              onChange={(e) => setSelectedMission(e.target.value)}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {MISSION_TYPES.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label} - {type.description}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-200 mb-2">
              Top N Recommendations
            </label>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-800 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-end">
            <button
              onClick={generateRecommendations}
              disabled={loading}
              className="w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Analyzing...' : 'Generate Recommendations'}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
            <span className="text-red-200">{error}</span>
          </div>
        </div>
      )}

      {/* Tab Content */}
      {activeTab === 'recommendations' && (
        <div className="space-y-4">
          {recommendations.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-300">No recommendations available. Generate recommendations to see ML-powered site analysis.</p>
            </div>
          ) : (
            recommendations.map((recommendation) => (
              <div key={recommendation.site_id} className="border border-gray-700 rounded-lg p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <div className="flex items-center space-x-3 mb-2">
                      <div className="flex items-center justify-center w-8 h-8 bg-purple-100 text-purple-800 rounded-full font-bold">
                        {recommendation.suitability_rank}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-gray-100">
                          Site {recommendation.site_id}
                        </h3>
                        <p className="text-sm text-gray-300">
                          {recommendation.coordinates[0].toFixed(2)}°, {recommendation.coordinates[1].toFixed(2)}°
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-4">
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${getCategoryColor(recommendation.suitability_category)}`}>
                        {recommendation.suitability_category}
                      </div>
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(recommendation.confidence)}`}>
                        Confidence: {(recommendation.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="flex items-center space-x-1">
                        <span className="text-sm text-gray-300">Score:</span>
                        <span className={`font-bold ${getScoreColor(recommendation.overall_score)}`}>
                          {(recommendation.overall_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-100 mb-3">Feature Analysis</h4>
                    <FeatureRadarChart scores={recommendation.feature_scores} />
                  </div>
                  
                  <div>
                    <div className="space-y-4">
                      {recommendation.recommendation_reasons.length > 0 && (
                        <div>
                          <h4 className="font-medium text-gray-100 mb-2 flex items-center">
                            <CheckCircle className="w-4 h-4 text-green-600 mr-2" />
                            Strengths
                          </h4>
                          <ul className="space-y-1">
                            {recommendation.recommendation_reasons.map((reason, idx) => (
                              <li key={idx} className="text-sm text-gray-300 flex items-start">
                                <span className="w-1.5 h-1.5 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {reason}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {recommendation.risk_factors.length > 0 && (
                        <div>
                          <h4 className="font-medium text-gray-100 mb-2 flex items-center">
                            <AlertTriangle className="w-4 h-4 text-orange-600 mr-2" />
                            Risk Factors
                          </h4>
                          <ul className="space-y-1">
                            {recommendation.risk_factors.map((risk, idx) => (
                              <li key={idx} className="text-sm text-gray-300 flex items-start">
                                <span className="w-1.5 h-1.5 bg-orange-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between text-sm text-gray-300">
                    <span>Cluster Assignment: #{recommendation.cluster_assignment}</span>
                    <span>Analysis for: {MISSION_TYPES.find(t => t.value === selectedMission)?.label}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'insights' && (
        <div className="space-y-6">
          {modelInsights ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-100 mb-2">Model Status</h3>
                  <p className="text-gray-300">
                    {modelInsights && 'model_status' in modelInsights && modelInsights.model_status === 'trained' ? '✅ Trained & Ready' : '❌ Not Trained'}
                  </p>
                </div>
                
                {modelInsights.performance_metrics?.r2_score && (
                  <div className="bg-gray-700 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-100 mb-2">R² Score</h3>
                    <p className="text-2xl font-bold text-gray-200">
                      {(modelInsights.performance_metrics.r2_score * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
                
                {modelInsights.clustering_info?.n_clusters && (
                  <div className="bg-gray-700 rounded-lg p-4">
                    <h3 className="font-semibold text-gray-100 mb-2">Clusters</h3>
                    <p className="text-2xl font-bold text-gray-200">
                      {modelInsights.clustering_info.n_clusters}
                    </p>
                  </div>
                )}
              </div>

              {Object.keys(modelInsights.feature_importance).length > 0 && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-100 mb-4">Feature Importance</h3>
                  <div className="space-y-3">
                    {Object.entries(modelInsights.feature_importance).map(([feature, importance]) => (
                      <div key={feature} className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-200 capitalize">
                          {feature.replace('_', ' ')}
                        </span>
                        <span className="text-sm font-bold text-gray-100">{importance}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {modelInsights.training_info?.samples && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-100 mb-2">Training Information</h3>
                  <div className="space-y-2 text-sm text-gray-300">
                    <p><strong>Samples:</strong> {modelInsights.training_info.samples}</p>
                    {modelInsights.training_info.training_date && (
                      <p><strong>Last Trained:</strong> {new Date(modelInsights.training_info.training_date).toLocaleDateString()}</p>
                    )}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-300">No model insights available. Train the models first.</p>
            </div>
          )}
        </div>
      )}

      {activeTab === 'training' && (
        <div className="space-y-6">
          <div className="bg-gray-700 rounded-lg p-6">
            <h3 className="font-semibold text-gray-100 mb-4">Model Training</h3>
            <p className="text-gray-300 mb-4">
              Train the machine learning models using historical Mars mission data to improve recommendation accuracy.
            </p>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-100 mb-2">Training Data</h4>
                <p className="text-sm text-gray-300">
                  Models will be trained on {SAMPLE_TRAINING_DATA.length} historical Mars mission sites with known outcomes.
                </p>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-100 mb-2">Model Features</h4>
                <div className="grid grid-cols-2 gap-2 text-sm text-gray-300">
                  <div>• Elevation</div>
                  <div>• Slope</div>
                  <div>• Aspect</div>
                  <div>• Roughness</div>
                  <div>• TRI</div>
                  <div>• Solar Exposure</div>
                  <div>• Resource Proximity</div>
                  <div>• Safety Score</div>
                </div>
              </div>
            </div>
            
            <button
              onClick={trainModels}
              disabled={trainingLoading}
              className="mt-4 px-6 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {trainingLoading ? 'Training Models...' : 'Train ML Models'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
