import React from "react";
import { getKpis } from "../api/client";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import type { DashboardResponse } from "../types";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function KpiDashboard() {
  const [data, setData] = React.useState<DashboardResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const datasetId = getDatasetId();

  React.useEffect(() => {
    let mounted = true;
    async function load() {
      if (!datasetId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await getKpis(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load KPIs.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const eda = data?.data?.eda_summary || {};
  const overview = eda.dataset_overview || { rows: 0, businesses: 0, sectors: 0 };
  const topSectors = eda.engagement_by_sector || [];
  const topBusinesses = eda.top_businesses_by_engagement_rate_followers || [];

  return (
    <div>
      <div className="text-xl font-semibold">KPI Dashboard</div>
      <div className="mt-1 text-sm text-slate-400">High-level metrics and leaderboards.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-4">
        <MetricCard label="Total Posts" value={overview.rows || 0} />
        <MetricCard label="Total Businesses" value={overview.businesses || 0} />
        <MetricCard label="Total Sectors" value={overview.sectors || 0} />
        <MetricCard label="Dataset ID" value={datasetId || "-"} />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        <DataTable title="Top Sectors (by avg engagement rate)" rows={topSectors} maxRows={15} />
        <DataTable title="Top Businesses (by avg engagement rate)" rows={topBusinesses} maxRows={20} />
      </div>
    </div>
  );
}

