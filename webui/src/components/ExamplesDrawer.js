import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
export default function ExamplesDrawer({ onSelectExample, isOpen, onClose }) {
    const [examples, setExamples] = useState([]);
    const [loading, setLoading] = useState(true);
    useEffect(() => {
        if (isOpen) {
            fetch('http://localhost:5000/api/v1/examples/rois')
                .then(res => res.json())
                .then(data => {
                setExamples(data);
                setLoading(false);
            })
                .catch(err => {
                console.error('Failed to load examples:', err);
                setLoading(false);
            });
        }
    }, [isOpen]);
    if (!isOpen)
        return null;
    return (_jsx("div", { className: "fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50", children: _jsxs("div", { className: "bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto", children: [_jsxs("div", { className: "flex justify-between items-center mb-4", children: [_jsx("h3", { className: "text-xl font-semibold", children: "Example Regions" }), _jsx("button", { onClick: onClose, className: "text-gray-400 hover:text-white text-2xl", children: "\u00D7" })] }), loading ? (_jsx("div", { className: "text-center py-8", children: "Loading examples..." })) : examples.length === 0 ? (_jsx("div", { className: "text-center py-8 text-gray-400", children: "No examples available" })) : (_jsx("div", { className: "space-y-3", children: examples.map(example => (_jsxs("div", { className: "bg-gray-700 rounded-lg p-4 hover:bg-gray-600 cursor-pointer transition-colors", onClick: () => {
                            onSelectExample(example);
                            onClose();
                        }, children: [_jsx("h4", { className: "font-semibold text-lg mb-1", children: example.name }), _jsx("p", { className: "text-sm text-gray-300", children: example.description }), _jsxs("div", { className: "text-xs text-gray-400 mt-2", children: ["Dataset: ", example.dataset.toUpperCase()] })] }, example.id))) }))] }) }));
}
