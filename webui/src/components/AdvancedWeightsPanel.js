import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
export default function AdvancedWeightsPanel({ weights, onChange, presetWeights }) {
    const [localWeights, setLocalWeights] = useState(weights);
    useEffect(() => {
        setLocalWeights(weights);
    }, [weights]);
    const updateWeight = (criterion, value) => {
        const newWeights = { ...localWeights, [criterion]: value };
        setLocalWeights(newWeights);
        onChange(newWeights);
    };
    const totalWeight = Object.values(localWeights).reduce((sum, w) => sum + w, 0);
    const isValid = Math.abs(totalWeight - 1.0) < 0.01;
    const criteria = [
        { key: 'slope', label: 'Slope Safety' },
        { key: 'roughness', label: 'Surface Roughness' },
        { key: 'elevation', label: 'Elevation' },
        { key: 'solar_exposure', label: 'Solar Exposure' },
        { key: 'science_value', label: 'Science Value' },
    ];
    return (_jsxs("div", { className: "mt-3 space-y-3", children: [criteria.map(({ key, label }) => (_jsxs("div", { children: [_jsxs("div", { className: "flex justify-between text-xs mb-1", children: [_jsx("span", { className: "text-gray-300", children: label }), _jsxs("span", { className: "text-gray-400", children: [((localWeights[key] || presetWeights[key] || 0) * 100).toFixed(0), "%"] })] }), _jsx("input", { type: "range", min: "0", max: "1", step: "0.01", value: localWeights[key] ?? presetWeights[key] ?? 0, onChange: (e) => updateWeight(key, parseFloat(e.target.value)), className: "w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer" })] }, key))), _jsx("div", { className: "pt-2 border-t border-gray-700", children: _jsxs("div", { className: "flex justify-between text-xs", children: [_jsxs("span", { className: isValid ? 'text-green-400' : 'text-yellow-400', children: ["Total Weight: ", (totalWeight * 100).toFixed(1), "%"] }), !isValid && (_jsx("span", { className: "text-yellow-400", children: "Should sum to 100%" }))] }) })] }));
}
