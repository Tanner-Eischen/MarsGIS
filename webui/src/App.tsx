import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import { GeoPlanProvider } from './context/GeoPlanContext'
import Dashboard from './pages/Dashboard'
import DataDownload from './pages/DataDownload'
import TerrainAnalysis from './pages/TerrainAnalysis'
import NavigationPlanning from './pages/NavigationPlanning'
import Visualization from './pages/Visualization'
import DecisionLab from './pages/DecisionLab'
import MissionScenarios from './pages/MissionScenarios'
import Projects from './pages/Projects'
import Validation from './pages/Validation'
import SolarAnalysis from './pages/SolarAnalysis'

function App() {
  return (
    <BrowserRouter
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <GeoPlanProvider>
        <Layout>
          <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/download" element={<DataDownload />} />
          <Route path="/analyze" element={<TerrainAnalysis />} />
          <Route path="/navigate" element={<NavigationPlanning />} />
          <Route path="/visualize" element={<Visualization />} />
          <Route path="/decision-lab" element={<DecisionLab />} />
          <Route path="/mission-scenarios" element={<MissionScenarios />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/validation" element={<Validation />} />
          <Route path="/solar-analysis" element={<SolarAnalysis />} />
          </Routes>
        </Layout>
      </GeoPlanProvider>
    </BrowserRouter>
  )
}

export default App




