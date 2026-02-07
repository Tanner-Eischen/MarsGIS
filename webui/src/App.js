import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import { GeoPlanProvider } from './context/GeoPlanContext';
import { OverlayLayerProvider } from './contexts/OverlayLayerContext';
import AnalysisDashboard from './pages/AnalysisDashboard';
import MissionControl from './pages/MissionControl';
import DecisionLab from './pages/DecisionLab';
import DataSettings from './pages/DataSettings';
function App() {
    return (_jsx(BrowserRouter, { future: {
            v7_startTransition: true,
            v7_relativeSplatPath: true,
        }, children: _jsx(GeoPlanProvider, { children: _jsx(OverlayLayerProvider, { children: _jsx(Layout, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(AnalysisDashboard, {}) }), _jsx(Route, { path: "/mission", element: _jsx(MissionControl, {}) }), _jsx(Route, { path: "/decision-lab", element: _jsx(DecisionLab, {}) }), _jsx(Route, { path: "/settings", element: _jsx(DataSettings, {}) })] }) }) }) }) }));
}
export default App;
