import React from "react";
import { getContentPerformance } from "../api/client";
import ChartCard from "../components/ChartCard";
import type { DashboardResponse } from "../types";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

function BarChartCard(props: { title: string; rows: any[]; xKey: string; yKey: string }) {
  return (
    <ChartCard title={props.title}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={props.rows}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey={props.xKey} tick={{ fill: "#cbd5e1", fontSize: 12 }} />
          <YAxis tick={{ fill: "#cbd5e1", fontSize: 12 }} />
          <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid #1f2937" }} />
          <Bar dataKey={props.yKey} fill="#60a5fa" />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export default function ContentPerformance() {
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
        const res = await getContentPerformance(datasetId);
        if (mounted) setData(res);
      } catch (e: any) {
        if (mounted) setError(e?.response?.data?.detail || e?.message || "Failed to load content performance.");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [datasetId]);

  const d = data?.data || {};
  const charts = [
    { title: "By Post Type", rows: d.engagement_by_post_type || [], xKey: "post_type", yKey: "avg_engagement_rate" },
    { title: "By Language", rows: d.engagement_by_language || [], xKey: "language", yKey: "avg_engagement_rate" },
    { title: "By CTA", rows: d.engagement_by_CTA_present || [], xKey: "CTA_present", yKey: "avg_engagement_rate" },
    { title: "By Promo", rows: d.engagement_by_promo_post || [], xKey: "promo_post", yKey: "avg_engagement_rate" },
    { title: "By Dialect", rows: d.engagement_by_arabic_dialect_style || [], xKey: "arabic_dialect_style", yKey: "avg_engagement_rate" },
    { title: "By Posting Hour", rows: d.engagement_by_posting_hour || [], xKey: "posting_hour", yKey: "avg_engagement_rate" }
  ];

  return (
    <div>
      <div className="text-xl font-semibold">Content Performance</div>
      <div className="mt-1 text-sm text-slate-400">Compare engagement rate across key dimensions.</div>

      {!datasetId ? <div className="mt-4 text-sm text-amber-200">Set a dataset_id (upload first).</div> : null}
      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        {charts.map((c) => (
          <BarChartCard key={c.title} title={c.title} rows={c.rows} xKey={c.xKey} yKey={c.yKey} />
        ))}
      </div>
    </div>
  );
}

