import { Route, Routes } from "react-router-dom";
import Layout from "../components/Layout";
import { GeoPlanProvider } from "../context/GeoPlanContext";
import { OverlayLayerProvider } from "../contexts/OverlayLayerContext";
import AnalysisDashboard from "./AnalysisDashboard";
import MissionControl from "./MissionControl";
import DecisionLab from "./DecisionLab";
import DataSettings from "./DataSettings";

export default function ExperimentalWorkspace() {
  return (
    <GeoPlanProvider>
      <OverlayLayerProvider>
        <div className="bg-amber-950 p-3 text-amber-100 text-sm">
          Experimental workspace · requires a configured local API. Solar and
          mission estimates are unvalidated.{" "}
          <a className="underline" href="/">
            Return to the verified Jezero analysis
          </a>
        </div>
        <Layout>
          <Routes>
            <Route path="analysis" element={<AnalysisDashboard />} />
            <Route path="mission" element={<MissionControl />} />
            <Route path="decision-lab" element={<DecisionLab />} />
            <Route path="settings" element={<DataSettings />} />
          </Routes>
        </Layout>
      </OverlayLayerProvider>
    </GeoPlanProvider>
  );
}
