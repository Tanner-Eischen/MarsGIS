import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';
export default function Layout({ children }) {
    const location = useLocation();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const navItems = [
        { path: '/', label: 'Dashboard' },
        { path: '/download', label: 'Download Data' },
        { path: '/analyze', label: 'Terrain Analysis' },
        { path: '/navigate', label: 'Navigation' },
        { path: '/visualize', label: 'Visualization' },
        { path: '/decision-lab', label: 'Decision Lab' },
        { path: '/mission-scenarios', label: 'Mission Scenarios' },
        { path: '/solar-analysis', label: 'Solar Analysis' },
        { path: '/projects', label: 'Projects' },
        { path: '/validation', label: 'Validation' },
    ];
    return (_jsxs("div", { className: "min-h-screen bg-gray-900 text-white", children: [_jsx("header", { className: "bg-gray-800 border-b border-gray-700", children: _jsxs("div", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8", children: [_jsxs("div", { className: "flex justify-between items-center h-16", children: [_jsxs("div", { className: "flex items-center", children: [_jsx("h1", { className: "text-2xl font-bold text-mars-orange", children: "MarsHab" }), _jsx("span", { className: "ml-2 text-sm text-gray-400 hidden sm:inline", children: "Mars Habitat Site Selection" })] }), _jsx("nav", { className: "hidden lg:flex space-x-2 xl:space-x-4", children: navItems.map((item) => (_jsx(Link, { to: item.path, className: `px-2 xl:px-3 py-2 rounded-md text-xs xl:text-sm font-medium transition-colors whitespace-nowrap ${location.pathname === item.path
                                            ? 'bg-mars-orange text-white'
                                            : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`, children: item.label }, item.path))) }), _jsx("button", { onClick: () => setMobileMenuOpen(!mobileMenuOpen), className: "lg:hidden p-2 rounded-md text-gray-300 hover:bg-gray-700 hover:text-white focus:outline-none focus:ring-2 focus:ring-mars-orange", "aria-label": "Toggle menu", children: _jsx("svg", { className: "h-6 w-6", fill: "none", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "2", viewBox: "0 0 24 24", stroke: "currentColor", children: mobileMenuOpen ? (_jsx("path", { d: "M6 18L18 6M6 6l12 12" })) : (_jsx("path", { d: "M4 6h16M4 12h16M4 18h16" })) }) })] }), mobileMenuOpen && (_jsx("nav", { className: "lg:hidden pb-4 space-y-1", children: navItems.map((item) => (_jsx(Link, { to: item.path, onClick: () => setMobileMenuOpen(false), className: `block px-3 py-2 rounded-md text-sm font-medium transition-colors ${location.pathname === item.path
                                    ? 'bg-mars-orange text-white'
                                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'}`, children: item.label }, item.path))) }))] }) }), _jsx("main", { className: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8", children: children })] }));
}
