import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export default function ExplainabilityPanel({ site, weights }) {
    if (!site) {
        return (_jsxs("div", { className: "p-4 bg-gray-800 rounded-lg border border-gray-700", children: [_jsx("h3", { className: "font-semibold mb-2", children: "Score Explanation" }), _jsx("p", { className: "text-sm text-gray-400", children: "Select a site to see score breakdown" })] }));
    }
    // Sort components by contribution (weighted value)
    const componentEntries = Object.entries(site.components)
        .map(([key, value]) => {
        const weight = weights[key] || 0;
        const contribution = value * weight;
        return { key, value, weight, contribution };
    })
        .sort((a, b) => b.contribution - a.contribution);
    const maxContribution = Math.max(...componentEntries.map(c => Math.abs(c.contribution)), 0.01);
    return (_jsxs("div", { className: "p-4 bg-gray-800 rounded-lg border border-gray-700", children: [_jsx("h3", { className: "font-semibold mb-3", children: "Score Explanation" }), _jsx("div", { className: "mb-4 pb-4 border-b border-gray-700", children: _jsxs("div", { className: "text-sm", children: [_jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("span", { className: "text-gray-400", children: "Site ID:" }), _jsx("span", { className: "font-semibold", children: site.site_id })] }), _jsxs("div", { className: "flex justify-between mb-1", children: [_jsx("span", { className: "text-gray-400", children: "Rank:" }), _jsxs("span", { className: "font-semibold", children: ["#", site.rank] })] }), _jsxs("div", { className: "flex justify-between", children: [_jsx("span", { className: "text-gray-400", children: "Total Score:" }), _jsxs("span", { className: "font-semibold text-blue-400", children: [(site.total_score * 100).toFixed(1), "%"] })] })] }) }), _jsxs("div", { className: "mb-4 pb-4 border-b border-gray-700", children: [_jsx("h4", { className: "text-sm font-semibold mb-2 text-gray-300", children: "Summary" }), _jsx("p", { className: "text-sm text-gray-300 leading-relaxed", children: site.explanation })] }), _jsxs("div", { children: [_jsx("h4", { className: "text-sm font-semibold mb-3 text-gray-300", children: "Component Contributions" }), _jsx("div", { className: "space-y-3", children: componentEntries.map(({ key, value, weight, contribution }) => {
                            const displayName = key
                                .replace(/_/g, ' ')
                                .replace(/\b\w/g, l => l.toUpperCase());
                            const barWidth = Math.abs(contribution) / maxContribution * 100;
                            const isPositive = contribution >= 0;
                            return (_jsxs("div", { className: "space-y-1", children: [_jsxs("div", { className: "flex justify-between text-xs", children: [_jsx("span", { className: "text-gray-400", children: displayName }), _jsxs("span", { className: "text-gray-300", children: [(contribution * 100).toFixed(1), "% contribution"] })] }), _jsxs("div", { className: "flex items-center gap-2", children: [_jsx("div", { className: "flex-1 h-2 bg-gray-700 rounded-full overflow-hidden", children: _jsx("div", { className: `h-full transition-all ${isPositive ? 'bg-green-500' : 'bg-red-500'}`, style: { width: `${barWidth}%` } }) }), _jsxs("div", { className: "text-xs text-gray-500 w-20 text-right", children: ["Value: ", typeof value === 'number' ? value.toFixed(2) : value] })] }), _jsxs("div", { className: "text-xs text-gray-500", children: ["Weight: ", (weight * 100).toFixed(0), "%"] })] }, key));
                        }) })] })] }));
}
