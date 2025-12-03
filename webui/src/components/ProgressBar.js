import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
export default function ProgressBar({ taskId, onComplete }) {
    const [progress, setProgress] = useState(0);
    const [stage, setStage] = useState('');
    const [message, setMessage] = useState('');
    const [estimatedTime, setEstimatedTime] = useState(null);
    const [isComplete, setIsComplete] = useState(false);
    const { isConnected, lastEvent } = useWebSocket({
        taskId,
        onProgress: (event) => {
            setProgress(event.progress);
            setStage(event.stage);
            setMessage(event.message);
            setEstimatedTime(event.estimated_seconds_remaining || null);
            if (event.progress >= 1.0) {
                setIsComplete(true);
                setTimeout(() => {
                    onComplete?.();
                }, 1000);
            }
        },
        onError: (error) => {
            console.error('Progress tracking error:', error);
        },
    });
    // Format time remaining
    const formatTimeRemaining = (seconds) => {
        if (seconds === null || seconds < 0)
            return '';
        if (seconds < 60)
            return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}m ${remainingSeconds}s`;
    };
    // Get stage display name
    const getStageName = (stage) => {
        const stageNames = {
            dem_loading: 'Loading DEM',
            terrain_metrics: 'Analyzing Terrain',
            criteria_extraction: 'Extracting Criteria',
            mcdm_evaluation: 'Evaluating Suitability',
            site_extraction: 'Extracting Sites',
            cost_map_generation: 'Generating Cost Map',
            pathfinding: 'Pathfinding',
            waypoint_generation: 'Generating Waypoints',
        };
        return stageNames[stage] || stage;
    };
    if (!taskId) {
        return null;
    }
    // Debug logging
    useEffect(() => {
        console.log('[ProgressBar] Task ID:', taskId, 'Connected:', isConnected, 'Last Event:', lastEvent);
    }, [taskId, isConnected, lastEvent]);
    if (isComplete) {
        return (_jsx("div", { className: "bg-green-900/50 border border-green-700 rounded-md p-4", children: _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "w-4 h-4 bg-green-500 rounded-full" }), _jsx("span", { className: "text-green-300 font-semibold", children: "Complete!" })] }) }));
    }
    return (_jsxs("div", { className: "bg-gray-800 border border-gray-700 rounded-md p-4", children: [!isConnected && (_jsx("div", { className: "mb-2 text-yellow-400 text-sm", children: "Connecting to progress stream..." })), _jsxs("div", { className: "mb-2 flex items-center justify-between", children: [_jsxs("div", { className: "flex-1", children: [_jsx("div", { className: "text-sm font-medium text-gray-300", children: getStageName(stage) || 'Processing...' }), message && (_jsx("div", { className: "text-xs text-gray-400 mt-1", children: message }))] }), _jsxs("div", { className: "text-sm text-gray-400 ml-4", children: [Math.round(progress * 100), "%"] })] }), _jsx("div", { className: "w-full bg-gray-700 rounded-full h-2.5 mb-2", children: _jsx("div", { className: "bg-blue-600 h-2.5 rounded-full transition-all duration-300", style: { width: `${progress * 100}%` } }) }), estimatedTime !== null && estimatedTime > 0 && (_jsxs("div", { className: "text-xs text-gray-400", children: ["Estimated time remaining: ", formatTimeRemaining(estimatedTime)] }))] }));
}
