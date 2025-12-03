import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import React, { useState } from 'react';
import { Layers, Settings, BarChart3, Download, Eye, Zap } from 'lucide-react';
export default function MultiResolutionFusion({ roi, className = '', onFusionComplete }) {
    const [datasetInfo, setDatasetInfo] = useState(null);
    const [fusionMethods, setFusionMethods] = useState(null);
    const [selectedDatasets, setSelectedDatasets] = useState([]);
    const [primaryDataset, setPrimaryDataset] = useState('mola');
    const [blendingMethod, setBlendingMethod] = useState('weighted_average');
    const [upsamplingMethod, setUpsamplingMethod] = useState('bilinear');
    const [downsamplingMethod, setDownsamplingMethod] = useState('average');
    const [targetResolution, setTargetResolution] = useState(undefined);
    const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
    const [edgePreservation, setEdgePreservation] = useState(true);
    const [noiseReduction, setNoiseReduction] = useState(true);
    const [isProcessing, setIsProcessing] = useState(false);
    const [response, setResponse] = useState(null);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [activeTab, setActiveTab] = useState('fusion');
    // Load dataset information on component mount
    React.useEffect(() => {
        loadDatasetInfo();
        loadFusionMethods();
    }, []);
    const loadDatasetInfo = async () => {
        try {
            const response = await fetch('/api/v1/fusion/info');
            const data = await response.json();
            setDatasetInfo(data);
        }
        catch (error) {
            console.error('Error loading dataset info:', error);
        }
    };
    const loadFusionMethods = async () => {
        try {
            const response = await fetch('/api/v1/fusion/methods');
            const data = await response.json();
            setFusionMethods(data);
        }
        catch (error) {
            console.error('Error loading fusion methods:', error);
        }
    };
    const handleDatasetToggle = (datasetId) => {
        setSelectedDatasets(prev => {
            if (prev.includes(datasetId)) {
                return prev.filter(id => id !== datasetId);
            }
            else {
                return [...prev, datasetId];
            }
        });
    };
    const performFusion = async () => {
        if (!roi || selectedDatasets.length === 0)
            return;
        setIsProcessing(true);
        setResponse(null);
        try {
            const request = {
                roi,
                datasets: selectedDatasets,
                primary_dataset: primaryDataset,
                blending_method: blendingMethod,
                upsampling_method: upsamplingMethod,
                downsampling_method: downsamplingMethod,
                target_resolution: targetResolution,
                confidence_threshold: confidenceThreshold,
                edge_preservation: edgePreservation,
                noise_reduction: noiseReduction,
            };
            const endpoint = activeTab === 'comparison' ? '/api/v1/fusion/compare' : '/api/v1/fusion/fuse';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });
            const data = await response.json();
            setResponse(data);
            if (data.success && onFusionComplete) {
                onFusionComplete(data);
            }
        }
        catch (error) {
            console.error('Error performing fusion:', error);
            setResponse({
                success: false,
                fused_data: { elevation: [], lat: [], lon: [], shape: [0, 0] },
                fusion_info: { method: '', target_resolution: 0, primary_dataset: '', input_datasets: [] },
                quality_metrics: { mean_elevation: 0, std_elevation: 0, elevation_range: 0, data_coverage: 0, roughness: 0 },
                dataset_info: {}
            });
        }
        finally {
            setIsProcessing(false);
        }
    };
    const getDatasetColor = (datasetId) => {
        const colors = {
            mola: 'bg-blue-500',
            hirise: 'bg-green-500',
            ctx: 'bg-purple-500'
        };
        return colors[datasetId] || 'bg-gray-500';
    };
    const getResolutionLabel = (resolution) => {
        if (resolution >= 100)
            return `${resolution}m`;
        if (resolution >= 1)
            return `${resolution}m`;
        return `${resolution * 100}cm`;
    };
    return (_jsxs("div", { className: `bg-white rounded-lg shadow-lg p-6 ${className}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx(Layers, { className: "w-6 h-6 text-blue-600" }), _jsx("h3", { className: "text-lg font-semibold text-gray-900", children: "Multi-Resolution Data Fusion" })] }), _jsxs("div", { className: "flex space-x-2", children: [_jsxs("button", { onClick: () => setActiveTab('fusion'), className: `px-3 py-1 rounded text-sm ${activeTab === 'fusion' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`, children: [_jsx(Zap, { className: "w-4 h-4 inline mr-1" }), "Fusion"] }), _jsxs("button", { onClick: () => setActiveTab('comparison'), className: `px-3 py-1 rounded text-sm ${activeTab === 'comparison' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`, children: [_jsx(BarChart3, { className: "w-4 h-4 inline mr-1" }), "Compare"] }), _jsxs("button", { onClick: () => setActiveTab('info'), className: `px-3 py-1 rounded text-sm ${activeTab === 'info' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`, children: [_jsx(Eye, { className: "w-4 h-4 inline mr-1" }), "Info"] })] })] }), activeTab === 'info' && datasetInfo && (_jsxs("div", { className: "space-y-4", children: [_jsx("h4", { className: "font-medium text-gray-900", children: "Available Datasets" }), _jsx("div", { className: "grid gap-3", children: Object.entries(datasetInfo.dataset_configs).map(([id, config]) => (_jsxs("div", { className: "p-3 bg-gray-50 rounded-lg", children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("span", { className: "font-medium text-gray-900", children: config.name }), _jsx("span", { className: `px-2 py-1 rounded text-xs text-white ${getDatasetColor(id)}`, children: getResolutionLabel(config.resolution) })] }), _jsxs("div", { className: "text-sm text-gray-600 space-y-1", children: [_jsxs("div", { children: ["Priority: ", config.priority] }), _jsxs("div", { children: ["Noise Level: ", config.noise_level] }), _jsxs("div", { children: ["Effective Range: ", config.effective_range[0], "m to ", config.effective_range[1], "m"] })] })] }, id))) })] })), (activeTab === 'fusion' || activeTab === 'comparison') && (_jsxs("div", { className: "space-y-4", children: [_jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: ["Select Datasets (", selectedDatasets.length, " selected)"] }), _jsx("div", { className: "grid grid-cols-3 gap-2", children: datasetInfo?.available_datasets.map(datasetId => {
                                    const config = datasetInfo.dataset_configs[datasetId];
                                    const isSelected = selectedDatasets.includes(datasetId);
                                    return (_jsxs("button", { onClick: () => handleDatasetToggle(datasetId), className: `p-3 rounded-lg border-2 transition-colors ${isSelected
                                            ? 'border-blue-500 bg-blue-50'
                                            : 'border-gray-200 bg-white hover:border-gray-300'}`, children: [_jsx("div", { className: "text-sm font-medium text-gray-900", children: config.name }), _jsx("div", { className: "text-xs text-gray-500", children: getResolutionLabel(config.resolution) })] }, datasetId));
                                }) })] }), selectedDatasets.length > 1 && (_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: "Primary Dataset" }), _jsx("select", { value: primaryDataset, onChange: (e) => setPrimaryDataset(e.target.value), className: "w-full p-2 border border-gray-300 rounded-lg", children: selectedDatasets.map(datasetId => (_jsx("option", { value: datasetId, children: datasetInfo?.dataset_configs[datasetId].name }, datasetId))) })] })), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: "Blending Method" }), _jsx("select", { value: blendingMethod, onChange: (e) => setBlendingMethod(e.target.value), className: "w-full p-2 border border-gray-300 rounded-lg", children: fusionMethods?.blending_methods.map(method => (_jsx("option", { value: method, children: method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) }, method))) })] }), _jsx("div", { children: _jsxs("button", { onClick: () => setShowAdvanced(!showAdvanced), className: "flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-800", children: [_jsx(Settings, { className: "w-4 h-4" }), _jsxs("span", { children: [showAdvanced ? 'Hide' : 'Show', " Advanced Settings"] })] }) }), showAdvanced && (_jsxs("div", { className: "space-y-4 p-4 bg-gray-50 rounded-lg", children: [_jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: "Upsampling Method" }), _jsx("select", { value: upsamplingMethod, onChange: (e) => setUpsamplingMethod(e.target.value), className: "w-full p-2 border border-gray-300 rounded-lg", children: fusionMethods?.upsampling_methods.map(method => (_jsx("option", { value: method, children: method.charAt(0).toUpperCase() + method.slice(1) }, method))) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: "Downsampling Method" }), _jsx("select", { value: downsamplingMethod, onChange: (e) => setDownsamplingMethod(e.target.value), className: "w-full p-2 border border-gray-300 rounded-lg", children: fusionMethods?.downsampling_methods.map(method => (_jsx("option", { value: method, children: method.charAt(0).toUpperCase() + method.slice(1) }, method))) })] }), _jsxs("div", { children: [_jsx("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: "Target Resolution (meters, optional)" }), _jsx("input", { type: "number", value: targetResolution || '', onChange: (e) => setTargetResolution(e.target.value ? parseFloat(e.target.value) : undefined), placeholder: "Leave empty to use primary dataset resolution", className: "w-full p-2 border border-gray-300 rounded-lg" })] }), _jsxs("div", { children: [_jsxs("label", { className: "block text-sm font-medium text-gray-900 mb-2", children: ["Confidence Threshold: ", confidenceThreshold] }), _jsx("input", { type: "range", min: "0", max: "1", step: "0.1", value: confidenceThreshold, onChange: (e) => setConfidenceThreshold(parseFloat(e.target.value)), className: "w-full" })] }), _jsxs("div", { className: "flex space-x-4", children: [_jsxs("label", { className: "flex items-center space-x-2", children: [_jsx("input", { type: "checkbox", checked: edgePreservation, onChange: (e) => setEdgePreservation(e.target.checked), className: "rounded" }), _jsx("span", { className: "text-sm text-gray-900", children: "Edge Preservation" })] }), _jsxs("label", { className: "flex items-center space-x-2", children: [_jsx("input", { type: "checkbox", checked: noiseReduction, onChange: (e) => setNoiseReduction(e.target.checked), className: "rounded" }), _jsx("span", { className: "text-sm text-gray-900", children: "Noise Reduction" })] })] })] })), _jsx("button", { onClick: performFusion, disabled: !roi || selectedDatasets.length === 0 || isProcessing, className: "w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2", children: isProcessing ? (_jsxs(_Fragment, { children: [_jsx("div", { className: "w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" }), _jsx("span", { children: "Processing..." })] })) : (_jsxs(_Fragment, { children: [_jsx(Zap, { className: "w-4 h-4" }), _jsx("span", { children: activeTab === 'comparison' ? 'Compare Datasets' : 'Fuse Datasets' })] })) }), response && (_jsxs("div", { className: `p-4 rounded-lg ${response.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsx("h4", { className: `font-medium ${response.success ? 'text-green-900' : 'text-red-900'}`, children: response.success ? 'Processing Complete' : 'Processing Failed' }), response.success && (_jsxs("button", { className: "flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800", children: [_jsx(Download, { className: "w-4 h-4" }), _jsx("span", { children: "Export" })] }))] }), response.success && (_jsxs("div", { className: "space-y-3", children: [_jsxs("div", { children: [_jsx("h5", { className: "text-sm font-medium text-gray-900 mb-1", children: "Fusion Information" }), _jsxs("div", { className: "text-sm text-gray-600 space-y-1", children: [_jsxs("div", { children: ["Method: ", response.fusion_info.method] }), _jsxs("div", { children: ["Target Resolution: ", response.fusion_info.target_resolution, "m"] }), _jsxs("div", { children: ["Primary Dataset: ", response.fusion_info.primary_dataset] }), _jsxs("div", { children: ["Input Datasets: ", response.fusion_info.input_datasets.join(', ')] })] })] }), _jsxs("div", { children: [_jsx("h5", { className: "text-sm font-medium text-gray-900 mb-1", children: "Quality Metrics" }), _jsxs("div", { className: "grid grid-cols-2 gap-2 text-sm text-gray-600", children: [_jsxs("div", { children: ["Mean Elevation: ", response.quality_metrics.mean_elevation.toFixed(2), "m"] }), _jsxs("div", { children: ["Std Elevation: ", response.quality_metrics.std_elevation.toFixed(2), "m"] }), _jsxs("div", { children: ["Elevation Range: ", response.quality_metrics.elevation_range.toFixed(2), "m"] }), _jsxs("div", { children: ["Data Coverage: ", (response.quality_metrics.data_coverage * 100).toFixed(1), "%"] }), _jsxs("div", { className: "col-span-2", children: ["Roughness: ", response.quality_metrics.roughness.toFixed(2)] })] })] }), Object.keys(response.dataset_info).length > 0 && (_jsxs("div", { children: [_jsx("h5", { className: "text-sm font-medium text-gray-900 mb-1", children: "Dataset Contributions" }), _jsx("div", { className: "text-sm text-gray-600", children: Object.entries(response.dataset_info).map(([id, info]) => (_jsxs("div", { children: [id, ": ", JSON.stringify(info)] }, id))) })] }))] }))] }))] }))] }));
}
