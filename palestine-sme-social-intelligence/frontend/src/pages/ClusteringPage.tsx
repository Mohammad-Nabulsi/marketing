import React from "react";
import { getClustering } from "../api/client";
import DataTable from "../components/DataTable";
import ChartCard from "../components/ChartCard";
import type { DashboardResponse } from "../types";
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function ClusteringPage() {
  const datasetId = getDatasetId();
  const [data, setData] = React.useState<DashboardResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let mounted = true;
    async function load() {
      if (!datasetId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await getClustering(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load clustering.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const postClusters = data?.data?.post_clusters || [];
  const bizClusters = data?.data?.business_clusters || [];
  const postPca = data?.data?.post_pca || [];

  return (
    <div>
      <div className="text-xl font-semibold">Clustering & PCA</div>
      <div className="mt-1 text-sm text-slate-400">Post clusters, business clusters, and a PCA scatter plot.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 grid grid-cols-1 gap-4">
        <ChartCard title="Post PCA (pca1 vs pca2)">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" dataKey="pca1" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <YAxis type="number" dataKey="pca2" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
              <Scatter data={postPca} fill="#a78bfa" />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>
        <DataTable title="Business Clusters" rows={bizClusters} maxRows={50} />
        <DataTable title="Post Clusters (preview)" rows={postClusters} maxRows={50} />
      </div>
    </div>
  );
}

