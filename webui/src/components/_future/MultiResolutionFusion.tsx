import React, { useState } from 'react';
import { Layers, Settings, BarChart3, Download, Eye, Zap } from 'lucide-react';
import { apiFetch } from '../../lib/apiBase';

interface FusionRequest {
  roi: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  datasets: string[];
  primary_dataset: string;
  blending_method: string;
  upsampling_method: string;
  downsampling_method: string;
  target_resolution?: number;
  confidence_threshold: number;
  edge_preservation: boolean;
  noise_reduction: boolean;
}

interface FusionResponse {
  success: boolean;
  fused_data: {
    elevation: number[][];
    lat: number[];
    lon: number[];
    shape: [number, number];
  };
  fusion_info: {
    method: string;
    target_resolution: number;
    primary_dataset: string;
    input_datasets: string[];
  };
  quality_metrics: {
    mean_elevation: number;
    std_elevation: number;
    elevation_range: number;
    data_coverage: number;
    roughness: number;
    inter_dataset_variance?: number;
  };
  dataset_info: Record<string, any>;
}

interface DatasetInfo {
  available_datasets: string[];
  dataset_configs: Record<string, {
    name: string;
    resolution: number;
    priority: number;
    noise_level: number;
    effective_range: [number, number];
  }>;
}

interface FusionMethods {
  blending_methods: string[];
  upsampling_methods: string[];
  downsampling_methods: string[];
}

interface MultiResolutionFusionProps {
  roi?: {
    lat_min: number;
    lat_max: number;
    lon_min: number;
    lon_max: number;
  };
  className?: string;
  onFusionComplete?: (response: FusionResponse) => void;
}

export default function MultiResolutionFusion({ 
  roi, 
  className = '', 
  onFusionComplete 
}: MultiResolutionFusionProps) {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [fusionMethods, setFusionMethods] = useState<FusionMethods | null>(null);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [primaryDataset, setPrimaryDataset] = useState<string>('mola_200m');
  const [blendingMethod, setBlendingMethod] = useState<string>('weighted_average');
  const [upsamplingMethod, setUpsamplingMethod] = useState<string>('bilinear');
  const [downsamplingMethod, setDownsamplingMethod] = useState<string>('average');
  const [targetResolution, setTargetResolution] = useState<number | undefined>(undefined);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);
  const [edgePreservation, setEdgePreservation] = useState<boolean>(true);
  const [noiseReduction, setNoiseReduction] = useState<boolean>(true);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [response, setResponse] = useState<FusionResponse | null>(null);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'fusion' | 'comparison' | 'info'>('fusion');

  React.useEffect(() => {
    loadDatasetInfo();
    loadFusionMethods();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const response = await apiFetch('/fusion/info');
      const data = await response.json();
      setDatasetInfo(data);
    } catch (error) {
      console.error('Error loading dataset info:', error);
    }
  };

  const loadFusionMethods = async () => {
    try {
      const response = await apiFetch('/fusion/methods');
      const data = await response.json();
      setFusionMethods(data);
    } catch (error) {
      console.error('Error loading fusion methods:', error);
    }
  };

  const handleDatasetToggle = (datasetId: string) => {
    setSelectedDatasets(prev => {
      if (prev.includes(datasetId)) {
        return prev.filter(id => id !== datasetId);
      } else {
        return [...prev, datasetId];
      }
    });
  };

  const performFusion = async () => {
    if (!roi || selectedDatasets.length === 0) return;

    setIsProcessing(true);
    setResponse(null);

    try {
      const request: FusionRequest = {
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

      const endpoint = activeTab === 'comparison' ? '/fusion/compare' : '/fusion/fuse';
      const response = await apiFetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`Fusion request failed with status ${response.status}`);
      }

      const data: FusionResponse = await response.json();
      setResponse(data);

      if (data.success && onFusionComplete) {
        onFusionComplete(data);
      }
    } catch (error) {
      console.error('Error performing fusion:', error);
      setResponse({
        success: false,
        fused_data: { elevation: [], lat: [], lon: [], shape: [0, 0] },
        fusion_info: { method: '', target_resolution: 0, primary_dataset: '', input_datasets: [] },
        quality_metrics: { mean_elevation: 0, std_elevation: 0, elevation_range: 0, data_coverage: 0, roughness: 0 },
        dataset_info: {}
      } as any);
    } finally {
      setIsProcessing(false);
    }
  };

  const getDatasetColor = (datasetId: string) => {
    const colors = {
      mola: 'bg-blue-500',
      hirise: 'bg-green-500',
      ctx: 'bg-purple-500'
    };
    return colors[datasetId as keyof typeof colors] || 'bg-gray-500';
  };

  const getResolutionLabel = (resolution: number) => {
    if (resolution >= 100) return `${resolution}m`;
    if (resolution >= 1) return `${resolution}m`;
    return `${resolution*100}cm`;
  };

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Layers className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Multi-Resolution Data Fusion</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('fusion')}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'fusion' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          >
            <Zap className="w-4 h-4 inline mr-1" />
            Fusion
          </button>
          <button
            onClick={() => setActiveTab('comparison')}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'comparison' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          >
            <BarChart3 className="w-4 h-4 inline mr-1" />
            Compare
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`px-3 py-1 rounded text-sm ${activeTab === 'info' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
          >
            <Eye className="w-4 h-4 inline mr-1" />
            Info
          </button>
        </div>
      </div>

      {activeTab === 'info' && datasetInfo && (
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900">Available Datasets</h4>
          <div className="grid gap-3">
            {Object.entries(datasetInfo.dataset_configs).map(([id, config]) => (
              <div key={id} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900">{config.name}</span>
                  <span className={`px-2 py-1 rounded text-xs text-white ${getDatasetColor(id)}`}>
                    {getResolutionLabel(config.resolution)}
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Priority: {config.priority}</div>
                  <div>Noise Level: {config.noise_level}</div>
                  <div>Effective Range: {config.effective_range[0]}m to {config.effective_range[1]}m</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(activeTab === 'fusion' || activeTab === 'comparison') && (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Select Datasets ({selectedDatasets.length} selected)
            </label>
            <div className="grid grid-cols-3 gap-2">
              {datasetInfo?.available_datasets.map(datasetId => {
                const config = datasetInfo.dataset_configs[datasetId];
                const isSelected = selectedDatasets.includes(datasetId);
                return (
                  <button
                    key={datasetId}
                    onClick={() => handleDatasetToggle(datasetId)}
                    className={`p-3 rounded-lg border-2 transition-colors ${
                      isSelected 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'border-gray-200 bg-white hover:border-gray-300'
                    }`}
                  >
                    <div className="text-sm font-medium text-gray-900">{config.name}</div>
                    <div className="text-xs text-gray-500">{getResolutionLabel(config.resolution)}</div>
                  </button>
                );
              })}
            </div>
          </div>

          {selectedDatasets.length > 1 && (
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">Primary Dataset</label>
              <select
                value={primaryDataset}
                onChange={(e) => setPrimaryDataset(e.target.value)}
                className="w-full p-2 border border-gray-300 rounded-lg"
              >
                {selectedDatasets.map(datasetId => (
                  <option key={datasetId} value={datasetId}>
                    {datasetInfo?.dataset_configs[datasetId].name}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-900 mb-2">Blending Method</label>
            <select
              value={blendingMethod}
              onChange={(e) => setBlendingMethod(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg"
            >
              {fusionMethods?.blending_methods.map(method => (
                <option key={method} value={method}>
                  {method.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center space-x-2 text-sm text-blue-600 hover:text-blue-800"
            >
              <Settings className="w-4 h-4" />
              <span>{showAdvanced ? 'Hide' : 'Show'} Advanced Settings</span>
            </button>
          </div>

          {showAdvanced && (
            <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Upsampling Method</label>
                <select
                  value={upsamplingMethod}
                  onChange={(e) => setUpsamplingMethod(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg"
                >
                  {fusionMethods?.upsampling_methods.map(method => (
                    <option key={method} value={method}>
                      {method.charAt(0).toUpperCase() + method.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Downsampling Method</label>
                <select
                  value={downsamplingMethod}
                  onChange={(e) => setDownsamplingMethod(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-lg"
                >
                  {fusionMethods?.downsampling_methods.map(method => (
                    <option key={method} value={method}>
                      {method.charAt(0).toUpperCase() + method.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Target Resolution (meters, optional)
                </label>
                <input
                  type="number"
                  value={targetResolution || ''}
                  onChange={(e) => setTargetResolution(e.target.value ? parseFloat(e.target.value) : undefined)}
                  placeholder="Leave empty to use primary dataset resolution"
                  className="w-full p-2 border border-gray-300 rounded-lg"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">
                  Confidence Threshold: {confidenceThreshold}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <div className="flex space-x-4">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={edgePreservation}
                    onChange={(e) => setEdgePreservation(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-900">Edge Preservation</span>
                </label>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={noiseReduction}
                    onChange={(e) => setNoiseReduction(e.target.checked)}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-900">Noise Reduction</span>
                </label>
              </div>
            </div>
          )}

          <button
            onClick={performFusion}
            disabled={!roi || selectedDatasets.length === 0 || isProcessing}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
          >
            {isProcessing ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Zap className="w-4 h-4" />
                <span>{activeTab === 'comparison' ? 'Compare Datasets' : 'Fuse Datasets'}</span>
              </>
            )}
          </button>

          {response && (
            <div className={`p-4 rounded-lg ${response.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'}`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className={`font-medium ${response.success ? 'text-green-900' : 'text-red-900'}`}>
                  {response.success ? 'Processing Complete' : 'Processing Failed'}
                </h4>
                {response.success && (
                  <button className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800">
                    <Download className="w-4 h-4" />
                    <span>Export</span>
                  </button>
                )}
              </div>

              {response.success && (
                <div className="space-y-3">
                  <div>
                    <h5 className="text-sm font-medium text-gray-900 mb-1">Fusion Information</h5>
                    <div className="text-sm text-gray-600 space-y-1">
                      <div>Method: {response.fusion_info.method}</div>
                      <div>Target Resolution: {response.fusion_info.target_resolution}m</div>
                      <div>Primary Dataset: {response.fusion_info.primary_dataset}</div>
                      <div>Input Datasets: {response.fusion_info.input_datasets.join(', ')}</div>
                    </div>
                  </div>

                  <div>
                    <h5 className="text-sm font-medium text-gray-900 mb-1">Quality Metrics</h5>
                    <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                      <div>Mean Elevation: {response.quality_metrics.mean_elevation.toFixed(2)}m</div>
                      <div>Std Elevation: {response.quality_metrics.std_elevation.toFixed(2)}m</div>
                      <div>Elevation Range: {response.quality_metrics.elevation_range.toFixed(2)}m</div>
                      <div>Data Coverage: {(response.quality_metrics.data_coverage * 100).toFixed(1)}%</div>
                      <div className="col-span-2">Roughness: {response.quality_metrics.roughness.toFixed(2)}</div>
                    </div>
                  </div>

                  {Object.keys(response.dataset_info).length > 0 && (
                    <div>
                      <h5 className="text-sm font-medium text-gray-900 mb-1">Dataset Contributions</h5>
                      <div className="text-sm text-gray-600">
                        {Object.entries(response.dataset_info).map(([id, info]) => (
                          <div key={id}>{id}: {JSON.stringify(info)}</div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
