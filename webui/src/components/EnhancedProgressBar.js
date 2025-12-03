import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
const STAGE_DEFINITIONS = {
    dem_loading: {
        name: 'Loading DEM Data',
        description: 'Downloading and processing Mars elevation data',
        typicalDuration: 30
    },
    terrain_metrics: {
        name: 'Analyzing Terrain',
        description: 'Calculating slope, aspect, roughness, and terrain ruggedness',
        typicalDuration: 45
    },
    criteria_extraction: {
        name: 'Extracting Criteria',
        description: 'Processing terrain data for site suitability analysis',
        typicalDuration: 20
    },
    mcdm_evaluation: {
        name: 'Evaluating Suitability',
        description: 'Running multi-criteria decision making algorithms',
        typicalDuration: 15
    },
    site_extraction: {
        name: 'Identifying Sites',
        description: 'Finding and ranking candidate construction sites',
        typicalDuration: 25
    }
};
export default function EnhancedProgressBar({ taskId, title = 'Analysis Progress', showDetails = true, onComplete, onError }) {
    const [stages, setStages] = useState([]);
    const [currentStage, setCurrentStage] = useState('');
    const [overallProgress, setOverallProgress] = useState(0);
    const [startTime, setStartTime] = useState(null);
    const [estimatedTotalTime, setEstimatedTotalTime] = useState(null);
    const { isConnected } = useWebSocket({
        taskId,
        onProgress: (event) => {
            console.log('[EnhancedProgressBar] Progress update:', event);
            setStages(prevStages => {
                const newStages = [...prevStages];
                const stageIndex = newStages.findIndex(s => s.key === event.stage);
                if (stageIndex >= 0) {
                    // Update existing stage
                    newStages[stageIndex] = {
                        ...newStages[stageIndex],
                        progress: event.progress,
                        message: event.message,
                        estimatedSecondsRemaining: event.estimated_seconds_remaining
                    };
                    // Mark previous stages as complete
                    for (let i = 0; i < stageIndex; i++) {
                        if (newStages[i].progress < 1.0) {
                            newStages[i].progress = 1.0;
                            newStages[i].duration = Date.now() - (newStages[i].startTime || Date.now());
                        }
                    }
                }
                else {
                    // Add new stage
                    const stageDef = STAGE_DEFINITIONS[event.stage];
                    newStages.push({
                        name: stageDef?.name || event.stage,
                        key: event.stage,
                        progress: event.progress,
                        message: event.message,
                        estimatedSecondsRemaining: event.estimated_seconds_remaining,
                        startTime: Date.now()
                    });
                }
                return newStages;
            });
            setCurrentStage(event.stage);
            setOverallProgress(event.progress);
            // Calculate total estimated time
            if (event.estimated_seconds_remaining && startTime) {
                const elapsed = (Date.now() - startTime) / 1000;
                const total = elapsed + event.estimated_seconds_remaining;
                setEstimatedTotalTime(total);
            }
            // Check for completion
            if (event.progress >= 1.0) {
                setTimeout(() => onComplete?.(), 500); // Small delay for visual feedback
            }
        },
        onError: (error) => {
            console.error('[EnhancedProgressBar] WebSocket error:', error);
            onError?.(error.message);
        }
    });
    useEffect(() => {
        if (taskId) {
            setStartTime(Date.now());
            setStages([]);
            setCurrentStage('');
            setOverallProgress(0);
            setEstimatedTotalTime(null);
        }
        else {
            setStartTime(null);
        }
    }, [taskId]);
    const formatTime = (seconds) => {
        if (seconds < 60)
            return `${Math.ceil(seconds)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.ceil(seconds % 60);
        return `${minutes}m ${remainingSeconds}s`;
    };
    const formatDuration = (ms) => {
        const seconds = ms / 1000;
        if (seconds < 60)
            return `${seconds.toFixed(1)}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = (seconds % 60).toFixed(1);
        return `${minutes}m ${remainingSeconds}s`;
    };
    if (!taskId) {
        return null;
    }
    const elapsedTime = startTime ? (Date.now() - startTime) / 1000 : 0;
    const remainingTime = estimatedTotalTime ? Math.max(0, estimatedTotalTime - elapsedTime) : null;
    return (_jsxs("div", { className: "bg-gray-800 rounded-lg p-6 border border-gray-700", children: [_jsxs("div", { className: "flex items-center justify-between mb-4", children: [_jsx("h3", { className: "text-lg font-semibold text-white", children: title }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: `w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}` }), _jsx("span", { className: "text-sm text-gray-400", children: isConnected ? 'Connected' : 'Disconnected' })] })] }), _jsxs("div", { className: "mb-6", children: [_jsxs("div", { className: "flex justify-between text-sm text-gray-300 mb-2", children: [_jsx("span", { children: "Overall Progress" }), _jsxs("span", { children: [(overallProgress * 100).toFixed(1), "%"] })] }), _jsx("div", { className: "w-full bg-gray-700 rounded-full h-3", children: _jsx("div", { className: "bg-gradient-to-r from-blue-500 to-green-500 h-3 rounded-full transition-all duration-300 ease-out", style: { width: `${overallProgress * 100}%` } }) }), remainingTime !== null && (_jsxs("div", { className: "flex justify-between text-xs text-gray-400 mt-1", children: [_jsxs("span", { children: ["Elapsed: ", formatTime(elapsedTime)] }), _jsxs("span", { children: ["Remaining: ", formatTime(remainingTime)] })] }))] }), showDetails && stages.length > 0 && (_jsxs("div", { className: "space-y-3", children: [_jsx("h4", { className: "text-sm font-medium text-gray-300 mb-3", children: "Analysis Stages" }), stages.map((stage) => {
                        const stageDef = STAGE_DEFINITIONS[stage.key];
                        const isActive = stage.key === currentStage;
                        const isCompleted = stage.progress >= 1.0;
                        return (_jsxs("div", { className: `p-3 rounded-lg border ${isActive ? 'border-blue-500 bg-blue-900/20' :
                                isCompleted ? 'border-green-500 bg-green-900/10' :
                                    'border-gray-600 bg-gray-750'}`, children: [_jsxs("div", { className: "flex items-center justify-between mb-2", children: [_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: `w-2 h-2 rounded-full ${isActive ? 'bg-blue-500 animate-pulse' :
                                                        isCompleted ? 'bg-green-500' :
                                                            'bg-gray-500'}` }), _jsx("span", { className: "text-sm font-medium text-white", children: stage.name })] }), _jsxs("span", { className: "text-xs text-gray-400", children: [(stage.progress * 100).toFixed(0), "%"] })] }), _jsx("div", { className: "w-full bg-gray-700 rounded-full h-2 mb-2", children: _jsx("div", { className: `h-2 rounded-full transition-all duration-300 ${isActive ? 'bg-blue-500' :
                                            isCompleted ? 'bg-green-500' :
                                                'bg-gray-500'}`, style: { width: `${stage.progress * 100}%` } }) }), _jsxs("div", { className: "flex justify-between items-center", children: [_jsx("p", { className: "text-xs text-gray-300", children: stage.message }), stage.duration && (_jsx("span", { className: "text-xs text-gray-400", children: formatDuration(stage.duration) }))] }), stage.estimatedSecondsRemaining && stage.progress < 1.0 && (_jsxs("div", { className: "text-xs text-gray-400 mt-1", children: ["~", formatTime(stage.estimatedSecondsRemaining), " remaining"] })), stageDef && (_jsx("p", { className: "text-xs text-gray-500 mt-1", children: stageDef.description }))] }, stage.key));
                    })] })), showDetails && (_jsx("div", { className: "mt-4 pt-4 border-t border-gray-700", children: _jsxs("details", { className: "text-xs text-gray-400", children: [_jsx("summary", { className: "cursor-pointer hover:text-gray-300", children: "Technical Details" }), _jsxs("div", { className: "mt-2 space-y-1", children: [_jsxs("div", { children: ["Task ID: ", _jsx("code", { className: "text-xs bg-gray-900 px-1 py-0.5 rounded", children: taskId })] }), _jsxs("div", { children: ["Connection: ", isConnected ? 'WebSocket Active' : 'WebSocket Disconnected'] }), _jsxs("div", { children: ["Stages Completed: ", stages.filter(s => s.progress >= 1.0).length, "/", stages.length] }), currentStage && _jsxs("div", { children: ["Current Stage: ", currentStage] })] })] }) }))] }));
}
