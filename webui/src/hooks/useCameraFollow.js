import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js/dist/plotly.min.js';
const DEFAULT_CONFIG = {
    distance: 0.01, // ~1km in degrees (approximate)
    height: 500, // meters above rover
    lookAheadDistance: 0.005, // ~500m ahead
};
/**
 * Hook for managing third-person camera that follows the rover
 */
export function useCameraFollow(roverPosition, isAnimating, plotRef, config = {}) {
    const cameraConfig = { ...DEFAULT_CONFIG, ...config };
    const lastUpdateRef = useRef(0);
    const THROTTLE_MS = 16; // ~60 FPS
    useEffect(() => {
        if (!roverPosition || !plotRef.current || !isAnimating) {
            return;
        }
        const now = performance.now();
        if (now - lastUpdateRef.current < THROTTLE_MS) {
            return;
        }
        lastUpdateRef.current = now;
        // Convert heading from degrees to radians
        const headingRad = (roverPosition.heading * Math.PI) / 180;
        // Calculate camera position behind and above rover
        // Note: For Mars coordinates, we use simplified spherical approximation
        const cameraOffsetX = -Math.sin(headingRad) * cameraConfig.distance;
        const cameraOffsetY = -Math.cos(headingRad) * cameraConfig.distance;
        const cameraX = roverPosition.lon + cameraOffsetX;
        const cameraY = roverPosition.lat + cameraOffsetY;
        const cameraZ = roverPosition.elevation + cameraConfig.height;
        // Calculate camera target (slightly ahead of rover)
        const targetOffsetX = Math.sin(headingRad) * cameraConfig.lookAheadDistance;
        const targetOffsetY = Math.cos(headingRad) * cameraConfig.lookAheadDistance;
        const targetX = roverPosition.lon + targetOffsetX;
        const targetY = roverPosition.lat + targetOffsetY;
        const targetZ = roverPosition.elevation;
        // Update camera using Plotly.relayout
        Plotly.relayout(plotRef.current, {
            scene: {
                camera: {
                    eye: { x: cameraX, y: cameraY, z: cameraZ },
                    center: { x: targetX, y: targetY, z: targetZ },
                    up: { x: 0, y: 0, z: 1 },
                },
            },
        });
    }, [roverPosition, isAnimating, plotRef, cameraConfig]);
}
