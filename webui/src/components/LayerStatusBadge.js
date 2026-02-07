import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function LayerStatusBadge({ status, dataset }) {
    const getStatusClasses = () => {
        switch (status) {
            case 'loading':
                return 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30';
            case 'cached':
                return 'bg-green-600/20 text-green-400 border-green-600/30';
            case 'error':
                return 'bg-red-600/20 text-red-400 border-red-600/30';
            case 'idle':
            default:
                return 'bg-gray-600/20 text-gray-400 border-gray-600/30';
        }
    };
    const getStatusText = () => {
        switch (status) {
            case 'loading':
                return 'Loading';
            case 'cached':
                return 'Cached';
            case 'error':
                return 'Error';
            case 'idle':
            default:
                return '';
        }
    };
    const getDatasetLabel = () => {
        switch (dataset) {
            case 'mola':
                return 'MOLA';
            case 'mola_200m':
                return 'MOLA200';
            case 'hirise':
                return 'HiRISE';
            case 'ctx':
                return 'CTX';
            default:
                return '';
        }
    };
    if (status === 'idle') {
        return null;
    }
    return (_jsxs("span", { className: `inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full border ${getStatusClasses()}`, children: [getDatasetLabel() && _jsx("span", { children: getDatasetLabel() }), _jsx("span", { children: getStatusText() })] }));
}
