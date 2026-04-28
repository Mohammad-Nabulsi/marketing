import React from "react";
import { getTrends } from "../api/client";
import ChartCard from "../components/ChartCard";
import DataTable from "../components/DataTable";
import type { DashboardResponse } from "../types";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function TrendsPage() {
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
        const res = await getTrends(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load trends.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const weekly = data?.data?.weekly_trends || [];
  const momentum = data?.data?.business_momentum || [];
  const forecast = data?.data?.forecast || [];
  const anomalies = data?.data?.anomalies || [];

  return (
    <div>
      <div className="text-xl font-semibold">Trends & Forecasting</div>
      <div className="mt-1 text-sm text-slate-400">Weekly trends, simple forecasts, and momentum per business.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 grid grid-cols-1 gap-4">
        <ChartCard title="Weekly Engagement Rate + Reels Share">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={weekly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="week" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <YAxis tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
              <Legend />
              <Line type="monotone" dataKey="avg_engagement_rate" stroke="#34d399" dot={false} />
              <Line type="monotone" dataKey="reels_share" stroke="#60a5fa" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Forecast (moving average)">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={forecast}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis dataKey="week" tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <YAxis tick={{ fill: "#cbd5e1", fontSize: 12 }} />
              <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
              <Legend />
              <Line type="monotone" dataKey="forecast_avg_engagement_rate" stroke="#a78bfa" dot={false} />
              <Line type="monotone" dataKey="forecast_avg_views_per_follower" stroke="#fbbf24" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <DataTable title="Business Momentum (top)" rows={momentum} maxRows={30} />
        <DataTable title="Anomalies (preview)" rows={anomalies} maxRows={50} />
      </div>
    </div>
  );
}

