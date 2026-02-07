import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Navigation, Database, FlaskConical } from 'lucide-react';
export default function Layout({ children }) {
    const location = useLocation();
    const navItems = [
        { path: '/', label: 'ANALYSIS', icon: _jsx(LayoutDashboard, { size: 18 }) },
        { path: '/mission', label: 'MISSION', icon: _jsx(Navigation, { size: 18 }) },
        { path: '/decision-lab', label: 'DECISION', icon: _jsx(FlaskConical, { size: 18 }) },
        { path: '/settings', label: 'DATA & CONFIG', icon: _jsx(Database, { size: 18 }) },
    ];
    return (_jsxs("div", { className: "h-screen flex flex-col bg-black text-white font-sans overflow-hidden", children: [_jsxs("header", { className: "h-16 bg-gray-900/80 backdrop-blur border-b border-cyan-900/30 flex items-center justify-between px-6 relative z-50", children: [_jsxs("div", { className: "flex items-center gap-3", children: [_jsx("div", { className: "w-8 h-8 bg-gradient-to-br from-mars-orange to-red-600 rounded-sm flex items-center justify-center shadow-lg shadow-orange-900/20", children: _jsx("span", { className: "font-bold text-white text-lg", children: "M" }) }), _jsxs("div", { children: [_jsxs("h1", { className: "text-xl font-bold tracking-widest text-white", children: ["MARS", _jsx("span", { className: "text-mars-orange", children: "GIS" })] }), _jsx("div", { className: "text-[0.6rem] text-cyan-500 tracking-[0.2em] uppercase", children: "Habitat Selection System" })] })] }), _jsx("nav", { className: "flex items-center gap-1 bg-gray-800/50 p-1 rounded-lg border border-gray-700/50", children: navItems.map((item) => {
                            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
                            return (_jsxs(Link, { to: item.path, className: `flex items-center gap-2 px-4 py-2 rounded-md text-sm font-bold transition-all duration-200 ${isActive
                                    ? 'bg-cyan-900/30 text-cyan-400 border border-cyan-500/30 shadow-[0_0_10px_rgba(6,182,212,0.15)]'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'}`, children: [item.icon, _jsx("span", { className: "tracking-wider", children: item.label })] }, item.path));
                        }) }), _jsxs("div", { className: "flex items-center gap-4", children: [_jsxs("div", { className: "flex flex-col items-end", children: [_jsx("span", { className: "text-xs text-cyan-400 font-mono", children: "SYS.ONLINE" }), _jsx("span", { className: "text-[0.6rem] text-gray-500 font-mono", children: "V.2.0.4" })] }), _jsx("div", { className: "w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]" })] })] }), _jsx("main", { className: "flex-1 overflow-hidden relative bg-grid-pattern", children: children })] }));
}
