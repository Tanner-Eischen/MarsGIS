import { useState, useCallback } from 'react';
import { Send, Brain, Settings, Info, AlertCircle, CheckCircle } from 'lucide-react';

interface AIQueryRequest {
  query: string;
  context?: {
    mission_type?: string;
    priority_criteria?: string[];
    constraints?: string[];
  };
}

interface AIQueryResponse {
  success: boolean;
  query: string;
  extracted_parameters?: {
    criteria_weights?: Record<string, number>;
    roi?: {
      min_lon: number;
      max_lon: number;
      min_lat: number;
      max_lat: number;
    };
    dataset_preferences?: {
      primary: string;
      resolution: string;
      coverage: string;
    };
    mission_constraints?: string[];
  };
  criteria_weights?: Record<string, number>;
  roi?: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  dataset?: string;
  confidence?: number;
  confidence_score?: number;
  explanation?: string;
  suggestions?: string[];
  error?: string;
  message?: string;
}

interface AIQueryInterfaceProps {
  onQueryProcessed: (parameters: AIQueryResponse['extracted_parameters']) => void;
  className?: string;
}

const EXAMPLE_QUERIES = [
  "Find me a flat site near water ice deposits with good solar exposure",
  "I need a smooth landing area in the northern hemisphere with low elevation",
  "Show me sites with gentle slopes near potential mineral resources",
  "Find elevated areas with good sunlight and low roughness for solar panels",
  "Locate flat terrain in Valles Marineris region for base construction"
];

export default function AIQueryInterface({ onQueryProcessed, className = '' }: AIQueryInterfaceProps) {
  const [query, setQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState<AIQueryResponse | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [selectedExample, setSelectedExample] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const processAIQuery = useCallback(async () => {
    if (!query.trim() || isProcessing) return;

    setIsProcessing(true);
    setResponse(null);

    try {
      const request: AIQueryRequest = {
        query: query.trim(),
        context: {
          mission_type: 'habitat_site_selection',
          priority_criteria: ['safety', 'resource_accessibility', 'sustainability'],
          constraints: ['martian_environment', 'technical_feasibility']
        }
      };

      const BASE = (import.meta as any).env?.VITE_API_URL || 'http://localhost:5000/api/v1';
      const response = await fetch(`${BASE}/ai-query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      const data: AIQueryResponse = await response.json();
      setResponse(data);

      if (data.success) {
        const weights = data.extracted_parameters?.criteria_weights || data.criteria_weights || {};
        const roiSrc = data.extracted_parameters?.roi || data.roi;
        const roiNorm = roiSrc
          ? {
              min_lat: (roiSrc as any).min_lat ?? (roiSrc as any).lat_min,
              max_lat: (roiSrc as any).max_lat ?? (roiSrc as any).lat_max,
              min_lon: (roiSrc as any).min_lon ?? (roiSrc as any).lon_min,
              max_lon: (roiSrc as any).max_lon ?? (roiSrc as any).lon_max,
            }
          : undefined;
        const datasetPref = data.extracted_parameters?.dataset_preferences || (data.dataset ? { primary: data.dataset } : undefined);
        onQueryProcessed({ criteria_weights: weights, roi: roiNorm, dataset_preferences: datasetPref } as any);
      }
    } catch (error) {
      setResponse({
        success: false,
        query,
        extracted_parameters: {},
        confidence_score: 0,
        explanation: 'Failed to process query due to network error',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setIsProcessing(false);
    }
  }, [query, isProcessing, onQueryProcessed]);

  const handleExampleClick = (example: string) => {
    setQuery(example);
    setSelectedExample(example);
    setShowExamples(false);
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceLabel = (score: number) => {
    if (score >= 0.8) return 'High Confidence';
    if (score >= 0.6) return 'Medium Confidence';
    return 'Low Confidence';
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Brain className="w-6 h-6 text-purple-400" />
          <h3 className="text-lg font-semibold text-gray-100">AI Query Assistant</h3>
        </div>
        <button
          onClick={() => setShowExamples(!showExamples)}
          className="flex items-center space-x-1 text-sm text-gray-300 hover:text-white"
        >
          <Settings className="w-4 h-4" />
          <span>Examples</span>
        </button>
      </div>

      <div className="space-y-4">
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Describe your ideal Mars habitat site in natural language..."
            className="w-full p-3 pr-12 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
            rows={3}
            disabled={isProcessing}
          />
          <button
            onClick={processAIQuery}
            disabled={!query.trim() || isProcessing}
            className="absolute bottom-3 right-3 p-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
          >
            {isProcessing ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>

        {showExamples && (
          <div className="bg-gray-700 rounded-lg p-4">
            <h4 className="text-sm font-medium text-gray-100 mb-2">Examples</h4>
            <div className="space-y-2">
              {EXAMPLE_QUERIES.map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className={`w-full text-left p-2 rounded text-sm bg-gray-800 hover:bg-gray-600 ${
                    selectedExample === example ? 'ring-1 ring-purple-500' : ''
                  }`}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        {response && (
          <div className={`rounded-lg p-4 ${response.success ? 'bg-green-900/30 border border-green-700' : 'bg-red-900/30 border border-red-700'}`}>
            <div className="flex items-start space-x-2">
              {response.success ? (
                <CheckCircle className="w-5 h-5 text-green-400 mt-0.5" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              )}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <h4 className={`text-sm font-medium ${response.success ? 'text-green-200' : 'text-red-200'}`}>
                    {response.success ? 'Query Processed Successfully' : 'Processing Failed'}
                  </h4>
                  {((response.confidence ?? 0) > 0 || (response.confidence_score ?? 0) > 0) && (
                    <span className={`text-xs font-medium ${getConfidenceColor((response.confidence ?? response.confidence_score ?? 0))}`}>
                      {getConfidenceLabel((response.confidence ?? response.confidence_score ?? 0))} ({Math.round(((response.confidence ?? response.confidence_score ?? 0) * 100))}%)
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-200 mb-3">{response.explanation}</p>
                <button onClick={() => setShowDetails(!showDetails)} className="text-xs text-gray-300 underline">
                  {showDetails ? 'Hide Details' : 'Show Details'}
                </button>

                {showDetails && response.success && (
                  <div className="space-y-3">
                    {(() => {
                      const weights = response.extracted_parameters?.criteria_weights || response.criteria_weights;
                      return weights && Object.keys(weights).length > 0;
                    })() && (
                      <div>
                        <h5 className="text-xs font-medium text-gray-200 mb-1">Extracted Criteria Weights:</h5>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(response.extracted_parameters?.criteria_weights || response.criteria_weights || {}).map(([criteria, weight]) => (
                            <div key={criteria} className="flex justify-between text-xs bg-gray-700 rounded px-2 py-1">
                              <span className="text-gray-300 capitalize">{criteria.replace('_', ' ')}:</span>
                              <span className="font-medium text-purple-400">{(weight as number).toFixed(2)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {(() => {
                      const roi = response.extracted_parameters?.roi || response.roi;
                      return !!roi;
                    })() && (
                      <div>
                        <h5 className="text-xs font-medium text-gray-200 mb-1">Region of Interest:</h5>
                        <div className="text-xs text-gray-300 bg-gray-700 rounded px-2 py-1">
                          {(() => {
                            const roi: any = response.extracted_parameters?.roi || response.roi;
                            const minLat = roi?.min_lat ?? roi?.lat_min;
                            const maxLat = roi?.max_lat ?? roi?.lat_max;
                            const minLon = roi?.min_lon ?? roi?.lon_min;
                            const maxLon = roi?.max_lon ?? roi?.lon_max;
                            return `Lat: ${minLat?.toFixed(2)}° to ${maxLat?.toFixed(2)}°\nLon: ${minLon?.toFixed(2)}° to ${maxLon?.toFixed(2)}°`;
                          })()}
                        </div>
                      </div>
                    )}

                    {(() => {
                      const dp = response.extracted_parameters?.dataset_preferences || (response.dataset ? { primary: response.dataset } : undefined);
                      return !!dp;
                    })() && (
                      <div>
                        <h5 className="text-xs font-medium text-gray-200 mb-1">Dataset Preferences:</h5>
                        <div className="text-xs text-gray-300 bg-gray-700 rounded px-2 py-1">
                          {(() => {
                            const dp: any = response.extracted_parameters?.dataset_preferences || (response.dataset ? { primary: response.dataset } : undefined);
                            const primary = dp?.primary ?? '';
                            const resolution = dp?.resolution ?? '';
                            const coverage = dp?.coverage ?? '';
                            return `Primary: ${primary}${resolution ? `\nResolution: ${resolution}` : ''}${coverage ? `\nCoverage: ${coverage}` : ''}`;
                          })()}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {showDetails && response.suggestions && response.suggestions.length > 0 && (
                  <div className="mt-3">
                    <h5 className="text-xs font-medium text-gray-200 mb-1 flex items-center">
                      <Info className="w-3 h-3 mr-1" />
                      Suggestions:
                    </h5>
                    <ul className="text-xs text-gray-300 space-y-1">
                      {response.suggestions.map((suggestion, index) => (
                        <li key={index} className="flex items-start">
                          <span className="mr-1">•</span>
                          <span>{suggestion}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}