import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import { GeoPlanProvider } from './context/GeoPlanContext'
import { OverlayLayerProvider } from './contexts/OverlayLayerContext'
import AnalysisDashboard from './pages/AnalysisDashboard'
import MissionControl from './pages/MissionControl'
import DecisionLab from './pages/DecisionLab'
import DataSettings from './pages/DataSettings'

function App() {
  return (
    <BrowserRouter
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <GeoPlanProvider>
        <OverlayLayerProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<AnalysisDashboard />} />
              <Route path="/mission" element={<MissionControl />} />
              <Route path="/decision-lab" element={<DecisionLab />} />
              <Route path="/settings" element={<DataSettings />} />
            </Routes>
          </Layout>
        </OverlayLayerProvider>
      </GeoPlanProvider>
    </BrowserRouter>
  )
}

export default App
