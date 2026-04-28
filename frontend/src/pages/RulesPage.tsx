import React from "react";
import { getRules } from "../api/client";
import DataTable from "../components/DataTable";
import type { DashboardResponse } from "../types";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function RulesPage() {
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
        const res = await getRules(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load rules.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const assoc = data?.data?.association_rules || [];
  const bv = data?.data?.business_value_rules || [];

  return (
    <div>
      <div className="text-xl font-semibold">Association Rules</div>
      <div className="mt-1 text-sm text-slate-400">Rules mined from transaction-style post attributes.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 grid grid-cols-1 gap-4">
        <div className="rounded-md border border-slate-800 bg-slate-900/40 p-4 text-sm text-slate-200">
          Look for consequents like <span className="font-mono">result=high_engagement</span> or <span className="font-mono">result=viral_post</span>. Higher lift can indicate stronger patterns.
        </div>
        <DataTable title="Business Value Rules (ranked)" rows={bv} maxRows={50} />
        <DataTable title="All Association Rules (preview)" rows={assoc} maxRows={50} />
      </div>
    </div>
  );
}

