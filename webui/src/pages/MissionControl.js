import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import NavigationPlanning from './NavigationPlanning';
import { Map, Flag, Activity } from 'lucide-react';
// Placeholder for Mission Scenarios until fully migrated/re-implemented inline
function MissionScenariosComponent() {
    return (_jsxs("div", { className: "glass-panel p-8 rounded-lg text-center text-gray-400", children: [_jsx(Flag, { className: "w-12 h-12 mx-auto mb-4 opacity-50" }), _jsx("h3", { className: "text-lg font-bold text-white mb-2", children: "MISSION_SCENARIOS" }), _jsx("p", { className: "text-sm", children: "Scenario builder is currently under maintenance." })] }));
}
export default function MissionControl() {
    const [activeTab, setActiveTab] = useState('planning');
    return (_jsxs("div", { className: "h-full flex flex-col bg-gray-900 text-white min-h-screen", children: [_jsxs("div", { className: "bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between", children: [_jsxs("div", { className: "flex items-center space-x-2", children: [_jsx(Activity, { className: "text-green-500" }), _jsx("h1", { className: "text-2xl font-bold tracking-wider", children: "MISSION_CONTROL" })] }), _jsxs("div", { className: "flex bg-gray-900 rounded-lg p-1 border border-gray-700", children: [_jsxs("button", { onClick: () => setActiveTab('planning'), className: `flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'planning'
                                    ? 'bg-green-600 text-white shadow-lg shadow-green-900/20'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-800'}`, children: [_jsx(Map, { className: "w-4 h-4 mr-2" }), "Route Planning"] }), _jsxs("button", { onClick: () => setActiveTab('scenarios'), className: `flex items-center px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'scenarios'
                                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/20'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-800'}`, children: [_jsx(Flag, { className: "w-4 h-4 mr-2" }), "Mission Scenarios"] })] })] }), _jsx("div", { className: "flex-1 p-6 overflow-auto", children: activeTab === 'planning' ? (_jsx("div", { className: "max-w-7xl mx-auto", children: _jsx(NavigationPlanning, {}) })) : (_jsx("div", { className: "max-w-7xl mx-auto", children: _jsx(MissionScenariosComponent, {}) })) })] }));
}
