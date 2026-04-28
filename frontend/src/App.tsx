import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import UploadPage from "./pages/UploadPage";
import KpiDashboard from "./pages/KpiDashboard";
import ContentPerformance from "./pages/ContentPerformance";
import ClusteringPage from "./pages/ClusteringPage";
import RulesPage from "./pages/RulesPage";
import TrendsPage from "./pages/TrendsPage";
import NetworkPage from "./pages/NetworkPage";
import RecommendationsPage from "./pages/RecommendationsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<UploadPage />} />
        <Route path="/kpis" element={<KpiDashboard />} />
        <Route path="/content" element={<ContentPerformance />} />
        <Route path="/clustering" element={<ClusteringPage />} />
        <Route path="/rules" element={<RulesPage />} />
        <Route path="/trends" element={<TrendsPage />} />
        <Route path="/network" element={<NetworkPage />} />
        <Route path="/recommendations" element={<RecommendationsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

