import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DataDownload from './pages/DataDownload'
import TerrainAnalysis from './pages/TerrainAnalysis'
import NavigationPlanning from './pages/NavigationPlanning'
import Visualization from './pages/Visualization'
import DecisionLab from './pages/DecisionLab'
import MissionScenarios from './pages/MissionScenarios'
import Projects from './pages/Projects'
import Validation from './pages/Validation'

function App() {
  return (
    <BrowserRouter>
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
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App




