import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState, useRef } from 'react';
import { MapContainer, ImageOverlay, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
// Fix for default marker icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
    iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});
function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => {
        if (bounds) {
            map.fitBounds(bounds, { padding: [20, 20] });
        }
    }, [map, bounds]);
    return null;
}
export default function SolarHeatmap({ roi, dataset = 'mola', solarPotentialMap, shape, loading = false, }) {
    const [heatmapImageUrl, setHeatmapImageUrl] = useState(null);
    const [imageBounds, setImageBounds] = useState(null);
    const [hoverValue, setHoverValue] = useState(null);
    const [hoverPosition, setHoverPosition] = useState(null);
    const heatmapImageUrlRef = useRef(null);
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    // Calculate center and bounds
    const center = [
        (roi.lat_min + roi.lat_max) / 2,
        (roi.lon_min + roi.lon_max) / 2,
    ];
    const bounds = [
        [roi.lat_min, roi.lon_min],
        [roi.lat_max, roi.lon_max],
    ];
    // Generate heatmap image from solar potential data
    useEffect(() => {
        if (!solarPotentialMap || !shape) {
            setHeatmapImageUrl(null);
            setImageBounds(null);
            return;
        }
        // Clean up previous image URL
        if (heatmapImageUrlRef.current) {
            URL.revokeObjectURL(heatmapImageUrlRef.current);
            heatmapImageUrlRef.current = null;
        }
        try {
            const canvas = document.createElement('canvas');
            canvas.width = shape.cols;
            canvas.height = shape.rows;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                console.error('Failed to get canvas context');
                return;
            }
            // Create image data
            const imageData = ctx.createImageData(shape.cols, shape.rows);
            const data = imageData.data;
            // Color scale: blue (low) -> yellow -> red (high)
            const getColor = (value) => {
                // Clamp value to [0, 1]
                const v = Math.max(0, Math.min(1, value));
                if (v < 0.5) {
                    // Blue to yellow (0 -> 0.5)
                    const t = v * 2;
                    return {
                        r: Math.floor(t * 255),
                        g: Math.floor(t * 255),
                        b: Math.floor((1 - t) * 255),
                        a: 200
                    };
                }
                else {
                    // Yellow to red (0.5 -> 1.0)
                    const t = (v - 0.5) * 2;
                    return {
                        r: 255,
                        g: Math.floor((1 - t) * 255),
                        b: 0,
                        a: 200
                    };
                }
            };
            // Fill image data
            for (let row = 0; row < shape.rows; row++) {
                for (let col = 0; col < shape.cols; col++) {
                    const value = solarPotentialMap[row]?.[col] ?? 0;
                    const color = getColor(value);
                    const idx = (row * shape.cols + col) * 4;
                    data[idx] = color.r;
                    data[idx + 1] = color.g;
                    data[idx + 2] = color.b;
                    data[idx + 3] = color.a;
                }
            }
            ctx.putImageData(imageData, 0, 0);
            // Convert to blob URL
            canvas.toBlob((blob) => {
                if (blob) {
                    const url = URL.createObjectURL(blob);
                    heatmapImageUrlRef.current = url;
                    setHeatmapImageUrl(url);
                    setImageBounds(bounds);
                    canvasRef.current = canvas;
                }
            }, 'image/png');
        }
        catch (error) {
            console.error('Failed to generate heatmap image:', error);
            setHeatmapImageUrl(null);
        }
    }, [solarPotentialMap, shape, bounds]);
    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (heatmapImageUrlRef.current) {
                URL.revokeObjectURL(heatmapImageUrlRef.current);
            }
        };
    }, []);
    // Handle mouse move for hover value
    const handleMouseMove = (e) => {
        if (!solarPotentialMap || !shape || !canvasRef.current || !containerRef.current)
            return;
        const map = e.target;
        const point = map.mouseEventToContainerPoint(e.originalEvent);
        const bounds = map.getBounds();
        // Convert screen coordinates to data coordinates
        const lat = bounds.getNorth() - (point.y / map.getSize().y) * (bounds.getNorth() - bounds.getSouth());
        const lon = bounds.getWest() + (point.x / map.getSize().x) * (bounds.getEast() - bounds.getWest());
        // Convert lat/lon to row/col
        const row = Math.floor((1 - (lat - roi.lat_min) / (roi.lat_max - roi.lat_min)) * shape.rows);
        const col = Math.floor((lon - roi.lon_min) / (roi.lon_max - roi.lon_min) * shape.cols);
        if (row >= 0 && row < shape.rows && col >= 0 && col < shape.cols) {
            const value = solarPotentialMap[row]?.[col] ?? 0;
            setHoverValue(value);
            // Calculate position relative to container
            const containerRect = containerRef.current.getBoundingClientRect();
            const mouseX = e.originalEvent.clientX - containerRect.left;
            const mouseY = e.originalEvent.clientY - containerRect.top;
            // Constrain tooltip to container bounds (with padding)
            const tooltipWidth = 180; // Approximate tooltip width
            const tooltipHeight = 40;
            const padding = 10;
            let tooltipX = mouseX;
            let tooltipY = mouseY - tooltipHeight - padding;
            // Constrain horizontally
            if (tooltipX - tooltipWidth / 2 < padding) {
                tooltipX = tooltipWidth / 2 + padding;
            }
            else if (tooltipX + tooltipWidth / 2 > containerRect.width - padding) {
                tooltipX = containerRect.width - tooltipWidth / 2 - padding;
            }
            // Constrain vertically
            if (tooltipY < padding) {
                tooltipY = mouseY + padding;
            }
            else if (tooltipY + tooltipHeight > containerRect.height - padding) {
                tooltipY = containerRect.height - tooltipHeight - padding;
            }
            setHoverPosition({ x: tooltipX, y: tooltipY });
        }
    };
    // const handleMouseLeave = () => {
    //   setHoverValue(null)
    //   setHoverPosition(null)
    // }
    if (loading) {
        return (_jsx("div", { className: "bg-gray-800 rounded-lg p-6", children: _jsx("div", { className: "h-[600px] w-full rounded-md flex items-center justify-center bg-gray-900", children: _jsx("p", { className: "text-gray-300", children: "Loading solar potential analysis..." }) }) }));
    }
    return (_jsxs("div", { className: "bg-gray-800 rounded-lg p-6", children: [_jsxs("div", { ref: containerRef, className: "h-[600px] w-full rounded-md overflow-hidden border border-gray-700 bg-gray-900 relative", style: { position: 'relative' }, children: [_jsxs(MapContainer, { center: center, zoom: 6, style: { height: '100%', width: '100%', backgroundColor: '#1a1a1a', minHeight: '600px' }, crs: L.CRS.EPSG4326, minZoom: 2, maxZoom: 15, zoomControl: true, worldCopyJump: false, children: [_jsx(FitBounds, { bounds: bounds }), heatmapImageUrl && imageBounds && (_jsx(ImageOverlay, { url: heatmapImageUrl, bounds: imageBounds, opacity: 0.7, eventHandlers: {
                                    mousemove: handleMouseMove
                                } }))] }, `solar-map-${roi.lat_min}-${roi.lat_max}-${roi.lon_min}-${roi.lon_max}-${dataset}`), hoverValue !== null && hoverPosition && (_jsxs("div", { className: "absolute pointer-events-none bg-gray-900 bg-opacity-90 text-white px-3 py-2 rounded shadow-lg text-sm z-50 whitespace-nowrap", style: {
                            left: `${hoverPosition.x}px`,
                            top: `${hoverPosition.y}px`,
                            transform: 'translateX(-50%)',
                            maxWidth: '90%',
                        }, children: ["Solar Potential: ", (hoverValue * 100).toFixed(1), "%"] }))] }), _jsxs("div", { className: "mt-4 flex items-center justify-between", children: [_jsx("div", { className: "text-sm text-gray-400", children: _jsx("p", { children: "Solar Potential Heatmap: Blue (low) \u2192 Yellow \u2192 Red (high)" }) }), _jsxs("div", { className: "flex items-center space-x-4", children: [_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: "w-4 h-4 bg-blue-500" }), _jsx("span", { className: "text-xs text-gray-400", children: "Low (0%)" })] }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: "w-4 h-4 bg-yellow-500" }), _jsx("span", { className: "text-xs text-gray-400", children: "Medium (50%)" })] }), _jsxs("div", { className: "flex items-center space-x-2", children: [_jsx("div", { className: "w-4 h-4 bg-red-500" }), _jsx("span", { className: "text-xs text-gray-400", children: "High (100%)" })] })] })] })] }));
}
