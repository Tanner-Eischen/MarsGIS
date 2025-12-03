import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import MissionLandingWizard from '../components/MissionLandingWizard';
import RoverTraverseWizard from '../components/RoverTraverseWizard';
import { useGeoPlan } from '../context/GeoPlanContext';
export default function MissionScenarios() {
    const [activeTab, setActiveTab] = useState('landing');
    const { landingSites } = useGeoPlan();
    return (_jsxs("div", { className: "space-y-6", children: [_jsx("h2", { className: "text-3xl font-bold", children: "Mission Scenarios" }), _jsxs("div", { className: "bg-gray-800 rounded-lg p-1 flex gap-2", children: [_jsx("button", { onClick: () => setActiveTab('landing'), className: `flex-1 px-4 py-2 rounded-md font-semibold transition-colors ${activeTab === 'landing'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`, children: "Landing Site Wizard" }), _jsx("button", { onClick: () => setActiveTab('traverse'), className: `flex-1 px-4 py-2 rounded-md font-semibold transition-colors ${activeTab === 'traverse'
                            ? 'bg-blue-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`, children: "Rover Traverse Wizard" })] }), _jsx("div", { className: "bg-gray-800 rounded-lg p-6 space-y-6", children: activeTab === 'landing' ? (_jsx(MissionLandingWizard, {})) : (_jsx(RoverTraverseWizard, {})) })] }));
}
