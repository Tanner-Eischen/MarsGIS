import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DataDownload from './pages/DataDownload'
import TerrainAnalysis from './pages/TerrainAnalysis'
import NavigationPlanning from './pages/NavigationPlanning'
import Visualization from './pages/Visualization'

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
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App




