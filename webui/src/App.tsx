import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import PortfolioShowcase from "./pages/PortfolioShowcase";

// API-dependent research tools are available locally, outside the public demo.
const ExperimentalWorkspace = import.meta.env.DEV
  ? lazy(() => import("./pages/ExperimentalWorkspace"))
  : null;

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<PortfolioShowcase />} />
        {ExperimentalWorkspace && (
          <Route
            path="/experimental/*"
            element={
              <Suspense
                fallback={<p role="status">Loading experimental workspace…</p>}
              >
                <ExperimentalWorkspace />
              </Suspense>
            }
          />
        )}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
