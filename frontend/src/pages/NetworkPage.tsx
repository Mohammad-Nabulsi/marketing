import React from "react";
import { getNetwork } from "../api/client";
import DataTable from "../components/DataTable";
import type { DashboardResponse } from "../types";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function NetworkPage() {
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
        const res = await getNetwork(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load network.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const nodes = data?.data?.nodes || [];
  const edges = data?.data?.edges || [];
  const summary = data?.data?.summary || {};

  return (
    <div>
      <div className="text-xl font-semibold">Network Analysis</div>
      <div className="mt-1 text-sm text-slate-400">Co-occurrence network from transaction items.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 rounded-md border border-slate-800 bg-slate-900/40 p-4">
        <div className="text-sm font-semibold">Network Summary</div>
        <pre className="mt-3 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-200">{JSON.stringify(summary, null, 2)}</pre>
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4">
        <DataTable title="Top Nodes (by pagerank)" rows={nodes} maxRows={50} />
        <DataTable title="Edges (by weight)" rows={edges} maxRows={50} />
      </div>
    </div>
  );
}

