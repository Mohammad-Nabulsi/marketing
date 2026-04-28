import React from "react";
import { getRecommendations } from "../api/client";
import DataTable from "../components/DataTable";
import type { DashboardResponse } from "../types";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

type Priority = "All" | "High" | "Medium" | "Low";

export default function RecommendationsPage() {
  const datasetId = getDatasetId();
  const [data, setData] = React.useState<DashboardResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [priority, setPriority] = React.useState<Priority>("All");

  React.useEffect(() => {
    let mounted = true;
    async function load() {
      if (!datasetId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await getRecommendations(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load recommendations.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const rows = (data?.data?.recommendations || []) as any[];
  const filtered = priority === "All" ? rows : rows.filter((r) => String(r.priority) === priority);

  return (
    <div>
      <div className="text-xl font-semibold">Recommendations</div>
      <div className="mt-1 text-sm text-slate-400">Explainable next steps with evidence and priority.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 flex flex-wrap gap-2">
        {(["All", "High", "Medium", "Low"] as Priority[]).map((p) => (
          <button
            key={p}
            onClick={() => setPriority(p)}
            className={[
              "rounded-md border px-3 py-2 text-sm",
              priority === p ? "border-slate-500 bg-slate-900 text-slate-100" : "border-slate-800 bg-slate-950 text-slate-300 hover:bg-slate-900/40"
            ].join(" ")}
          >
            {p}
          </button>
        ))}
      </div>

      <div className="mt-4">
        <DataTable title="Recommendations" rows={filtered} maxRows={80} />
      </div>
    </div>
  );
}

