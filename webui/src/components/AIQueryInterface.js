import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useCallback } from 'react';
import { Send, Brain, Settings, Info, AlertCircle, CheckCircle } from 'lucide-react';
const EXAMPLE_QUERIES = [
    "Find me a flat site near water ice deposits with good solar exposure",
    "I need a smooth landing area in the northern hemisphere with low elevation",
    "Show me sites with gentle slopes near potential mineral resources",
    "Find elevated areas with good sunlight and low roughness for solar panels",
    "Locate flat terrain in Valles Marineris region for base construction"
];
export default function AIQueryInterface({ onQueryProcessed, className = '' }) {
    const [query, setQuery] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [response, setResponse] = useState(null);
    const [showExamples, setShowExamples] = useState(false);
    const [selectedExample, setSelectedExample] = useState(null);
    const [showDetails, setShowDetails] = useState(false);
    const processAIQuery = useCallback(async () => {
        if (!query.trim() || isProcessing)
            return;
        setIsProcessing(true);
        setResponse(null);
        try {
            const request = {
                query: query.trim(),
                context: {
                    mission_type: 'habitat_site_selection',
                    priority_criteria: ['safety', 'resource_accessibility', 'sustainability'],
                    constraints: ['martian_environment', 'technical_feasibility']
                }
            };
            const BASE = import.meta.env?.VITE_API_URL || 'http://localhost:5000/api/v1';
            const response = await fetch(`${BASE}/ai-query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });
            const data = await response.json();
            setResponse(data);
            if (data.success) {
                const weights = data.extracted_parameters?.criteria_weights || data.criteria_weights || {};
                const roiSrc = data.extracted_parameters?.roi || data.roi;
                const roiNorm = roiSrc
                    ? {
                        min_lat: roiSrc.min_lat ?? roiSrc.lat_min,
                        max_lat: roiSrc.max_lat ?? roiSrc.lat_max,
                        min_lon: roiSrc.min_lon ?? roiSrc.lon_min,
                        max_lon: roiSrc.max_lon ?? roiSrc.lon_max,
                    }
                    : undefined;
                const datasetPref = data.extracted_parameters?.dataset_preferences || (data.dataset ? { primary: data.dataset } : undefined);
                onQueryProcessed({ criteria_weights: weights, roi: roiNorm, dataset_preferences: datasetPref });
            }
        }
        catch (error) {
            setResponse({
                success: false,
                query,
                extracted_parameters: {},
                confidence_score: 0,
                explanation: 'Failed to process query due to network error',
                error: error instanceof Error ? error.message : 'Unknown error'
            });
        }
        finally {
            setIsProcessing(false);
        }
    }, [query, isProcessing, onQueryProcessed]);
    const handleExampleClick = (example) => {
        setQuery(example);
        setSelectedExample(example);
        setShowExamples(false);
    };
    const getConfidenceColor = (score) => {
        if (score >= 0.8)
            return 'text-green-600';
        if (score >= 0.6)
            return 'text-yellow-600';
        return 'text-red-600';
    };
    const getConfidenceLabel = (score) => {
        if (score >= 0.8)
            return 'High Confidence';
        if (score >= 0.6)
            return 'Medium Confidence';
        return 'Low Confidence';
    };
    return (_jsxs("div", { className: `bg-gray-800 rounded-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx(Brain, { className: "w-6 h-6 text-purple-400" }), _jsx("h3", { className: "text-lg font-semibold text-gray-100", children: "AI Query Assistant" })] }), _jsxs("button", { onClick: () => setShowExamples(!showExamples), className: "flex items-center space-x-1 text-sm text-gray-300 hover:text-white", children: [_jsx(Settings, { className: "w-4 h-4" }), _jsx("span", { children: "Examples" })] })] }), _jsxs("div", { className: "space-y-4", children: [_jsxs("div", { className: "relative", children: [_jsx("textarea", { value: query, onChange: (e) => setQuery(e.target.value), placeholder: "Describe your ideal Mars habitat site in natural language...", className: "w-full p-3 pr-12 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none", rows: 3, disabled: isProcessing }), _jsx("button", { onClick: processAIQuery, disabled: !query.trim() || isProcessing, className: "absolute bottom-3 right-3 p-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-500 disabled:cursor-not-allowed", children: isProcessing ? (_jsx("div", { className: "w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" })) : (_jsx(Send, { className: "w-4 h-4" })) })] }), showExamples && (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4", children: [_jsx("h4", { className: "text-sm font-medium text-gray-100 mb-2", children: "Examples" }), _jsx("div", { className: "space-y-2", children: EXAMPLE_QUERIES.map((example, index) => (_jsx("button", { onClick: () => handleExampleClick(example), className: `w-full text-left p-2 rounded text-sm bg-gray-800 hover:bg-gray-600 ${selectedExample === example ? 'ring-1 ring-purple-500' : ''}`, children: example }, index))) })] })), response && (_jsx("div", { className: `rounded-lg p-4 ${response.success ? 'bg-green-900/30 border border-green-700' : 'bg-red-900/30 border border-red-700'}`, children: _jsxs("div", { className: "flex items-start space-x-2", children: [response.success ? (_jsx(CheckCircle, { className: "w-5 h-5 text-green-400 mt-0.5" })) : (_jsx(AlertCircle, { className: "w-5 h-5 text-red-400 mt-0.5" })), _jsxs("div", { className: "flex-1", children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("h4", { className: `text-sm font-medium ${response.success ? 'text-green-200' : 'text-red-200'}`, children: response.success ? 'Query Processed Successfully' : 'Processing Failed' }), ((response.confidence ?? 0) > 0 || (response.confidence_score ?? 0) > 0) && (_jsxs("span", { className: `text-xs font-medium ${getConfidenceColor((response.confidence ?? response.confidence_score ?? 0))}`, children: [getConfidenceLabel((response.confidence ?? response.confidence_score ?? 0)), " (", Math.round(((response.confidence ?? response.confidence_score ?? 0) * 100)), "%)"] }))] }), _jsx("p", { className: "text-sm text-gray-200 mb-3", children: response.explanation }), _jsx("button", { onClick: () => setShowDetails(!showDetails), className: "text-xs text-gray-300 underline", children: showDetails ? 'Hide Details' : 'Show Details' }), showDetails && response.success && (_jsxs("div", { className: "space-y-3", children: [(() => {
                                                    const weights = response.extracted_parameters?.criteria_weights || response.criteria_weights;
                                                    return weights && Object.keys(weights).length > 0;
                                                })() && (_jsxs("div", { children: [_jsx("h5", { className: "text-xs font-medium text-gray-200 mb-1", children: "Extracted Criteria Weights:" }), _jsx("div", { className: "grid grid-cols-2 gap-2", children: Object.entries(response.extracted_parameters?.criteria_weights || response.criteria_weights || {}).map(([criteria, weight]) => (_jsxs("div", { className: "flex justify-between text-xs bg-gray-700 rounded px-2 py-1", children: [_jsxs("span", { className: "text-gray-300 capitalize", children: [criteria.replace('_', ' '), ":"] }), _jsx("span", { className: "font-medium text-purple-400", children: weight.toFixed(2) })] }, criteria))) })] })), (() => {
                                                    const roi = response.extracted_parameters?.roi || response.roi;
                                                    return !!roi;
                                                })() && (_jsxs("div", { children: [_jsx("h5", { className: "text-xs font-medium text-gray-200 mb-1", children: "Region of Interest:" }), _jsx("div", { className: "text-xs text-gray-300 bg-gray-700 rounded px-2 py-1", children: (() => {
                                                                const roi = response.extracted_parameters?.roi || response.roi;
                                                                const minLat = roi?.min_lat ?? roi?.lat_min;
                                                                const maxLat = roi?.max_lat ?? roi?.lat_max;
                                                                const minLon = roi?.min_lon ?? roi?.lon_min;
                                                                const maxLon = roi?.max_lon ?? roi?.lon_max;
                                                                return `Lat: ${minLat?.toFixed(2)}° to ${maxLat?.toFixed(2)}°\nLon: ${minLon?.toFixed(2)}° to ${maxLon?.toFixed(2)}°`;
                                                            })() })] })), (() => {
                                                    const dp = response.extracted_parameters?.dataset_preferences || (response.dataset ? { primary: response.dataset } : undefined);
                                                    return !!dp;
                                                })() && (_jsxs("div", { children: [_jsx("h5", { className: "text-xs font-medium text-gray-200 mb-1", children: "Dataset Preferences:" }), _jsx("div", { className: "text-xs text-gray-300 bg-gray-700 rounded px-2 py-1", children: (() => {
                                                                const dp = response.extracted_parameters?.dataset_preferences || (response.dataset ? { primary: response.dataset } : undefined);
                                                                const primary = dp?.primary ?? '';
                                                                const resolution = dp?.resolution ?? '';
                                                                const coverage = dp?.coverage ?? '';
                                                                return `Primary: ${primary}${resolution ? `\nResolution: ${resolution}` : ''}${coverage ? `\nCoverage: ${coverage}` : ''}`;
                                                            })() })] }))] })), showDetails && response.suggestions && response.suggestions.length > 0 && (_jsxs("div", { className: "mt-3", children: [_jsxs("h5", { className: "text-xs font-medium text-gray-200 mb-1 flex items-center", children: [_jsx(Info, { className: "w-3 h-3 mr-1" }), "Suggestions:"] }), _jsx("ul", { className: "text-xs text-gray-300 space-y-1", children: response.suggestions.map((suggestion, index) => (_jsxs("li", { className: "flex items-start", children: [_jsx("span", { className: "mr-1", children: "\u2022" }), _jsx("span", { children: suggestion })] }, index))) })] }))] })] }) }))] })] }));
}
