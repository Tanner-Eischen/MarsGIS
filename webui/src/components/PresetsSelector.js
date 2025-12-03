import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function PresetsSelector({ presets, selected, onSelect }) {
    return (_jsxs("div", { className: "p-4 border-b border-gray-700", children: [_jsx("h3", { className: "font-semibold mb-3", children: "Preset Selection" }), _jsx("div", { className: "space-y-2", children: presets.map((preset) => (_jsxs("button", { onClick: () => onSelect(preset.id), className: `w-full text-left p-3 rounded border-2 transition-colors ${selected === preset.id
                        ? 'border-blue-500 bg-blue-900/30'
                        : 'border-gray-700 bg-gray-700/50 hover:bg-gray-700'}`, children: [_jsx("div", { className: "font-semibold text-sm", children: preset.name }), _jsx("div", { className: "text-xs text-gray-400 mt-1", children: preset.description })] }, preset.id))) })] }));
}
