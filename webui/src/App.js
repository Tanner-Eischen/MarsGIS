import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import { GeoPlanProvider } from './context/GeoPlanContext';
import Dashboard from './pages/Dashboard';
import DataDownload from './pages/DataDownload';
import TerrainAnalysis from './pages/TerrainAnalysis';
import NavigationPlanning from './pages/NavigationPlanning';
import Visualization from './pages/Visualization';
import DecisionLab from './pages/DecisionLab';
import MissionScenarios from './pages/MissionScenarios';
import Projects from './pages/Projects';
import Validation from './pages/Validation';
import SolarAnalysis from './pages/SolarAnalysis';
function App() {
    return (_jsx(BrowserRouter, { future: {
            v7_startTransition: true,
            v7_relativeSplatPath: true,
        }, children: _jsx(GeoPlanProvider, { children: _jsx(Layout, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Dashboard, {}) }), _jsx(Route, { path: "/download", element: _jsx(DataDownload, {}) }), _jsx(Route, { path: "/analyze", element: _jsx(TerrainAnalysis, {}) }), _jsx(Route, { path: "/navigate", element: _jsx(NavigationPlanning, {}) }), _jsx(Route, { path: "/visualize", element: _jsx(Visualization, {}) }), _jsx(Route, { path: "/decision-lab", element: _jsx(DecisionLab, {}) }), _jsx(Route, { path: "/mission-scenarios", element: _jsx(MissionScenarios, {}) }), _jsx(Route, { path: "/projects", element: _jsx(Projects, {}) }), _jsx(Route, { path: "/validation", element: _jsx(Validation, {}) }), _jsx(Route, { path: "/solar-analysis", element: _jsx(SolarAnalysis, {}) })] }) }) }) }));
}
export default App;
