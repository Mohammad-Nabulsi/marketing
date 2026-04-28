import React from "react";
import { runPipeline, uploadDataset } from "../api/client";
import type { PipelineSummary, UploadResponse } from "../types";

function getDatasetId() {
  return localStorage.getItem("dataset_id") || "";
}

export default function UploadPage() {
  const [file, setFile] = React.useState<File | null>(null);
  const [uploadRes, setUploadRes] = React.useState<UploadResponse | null>(null);
  const [pipelineRes, setPipelineRes] = React.useState<PipelineSummary | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  async function onUpload() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setUploadRes(null);
    setPipelineRes(null);
    try {
      const res = await uploadDataset(file);
      localStorage.setItem("dataset_id", res.dataset_id);
      setUploadRes(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  }

  async function onRunPipeline() {
    const datasetId = getDatasetId();
    if (!datasetId) return;
    setLoading(true);
    setError(null);
    setPipelineRes(null);
    try {
      const res = await runPipeline(datasetId);
      setPipelineRes(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Pipeline failed.");
    } finally {
      setLoading(false);
    }
  }

  const datasetId = getDatasetId();

  return (
    <div>
      <div className="text-xl font-semibold">Upload & Data Quality</div>
      <div className="mt-1 text-sm text-slate-400">Upload a CSV dataset, view validation report, then run the full pipeline.</div>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        <div className="rounded-md border border-slate-800 bg-slate-900/40 p-4">
          <div className="text-sm font-semibold">Upload CSV</div>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mt-3 block w-full rounded-md border border-slate-800 bg-slate-950 px-3 py-2 text-sm"
          />
          <button
            onClick={onUpload}
            disabled={!file || loading}
            className="mt-3 rounded-md bg-slate-200 px-3 py-2 text-sm font-semibold text-slate-950 disabled:opacity-50"
          >
            Upload
          </button>
          <div className="mt-3 text-xs text-slate-500">Saved `dataset_id` will appear in the sidebar.</div>
        </div>

        <div className="rounded-md border border-slate-800 bg-slate-900/40 p-4">
          <div className="text-sm font-semibold">Run Full Pipeline</div>
          <div className="mt-2 text-sm text-slate-300">
            Current dataset_id: <span className="font-mono text-slate-100">{datasetId || "(none)"}</span>
          </div>
          <button
            onClick={onRunPipeline}
            disabled={!datasetId || loading}
            className="mt-3 rounded-md bg-emerald-400 px-3 py-2 text-sm font-semibold text-slate-950 disabled:opacity-50"
          >
            Run Full Pipeline
          </button>
          <div className="mt-3 text-xs text-slate-500">Outputs will be saved under `backend/storage/outputs/{datasetId}/`.</div>
        </div>
      </div>

      {loading ? <div className="mt-4 text-sm text-slate-300">Loading…</div> : null}
      {error ? <div className="mt-4 rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-200">{error}</div> : null}

      {uploadRes ? (
        <div className="mt-6 rounded-md border border-slate-800 bg-slate-900/40 p-4">
          <div className="text-sm font-semibold">Validation Report</div>
          <div className="mt-2 text-sm">
            Status:{" "}
            <span className={uploadRes.validation_report.ok ? "text-emerald-300" : "text-amber-300"}>
              {uploadRes.validation_report.ok ? "OK" : "Has issues"}
            </span>
          </div>
          <pre className="mt-3 overflow-auto rounded-md bg-slate-950 p-3 text-xs text-slate-200">
            {JSON.stringify(uploadRes.validation_report, null, 2)}
          </pre>
        </div>
      ) : null}

      {pipelineRes ? (
        <div className="mt-6 rounded-md border border-slate-800 bg-slate-900/40 p-4">
          <div className="text-sm font-semibold">Pipeline Summary</div>
          <div className="mt-2 text-sm text-slate-300">{pipelineRes.message}</div>
          <div className="mt-3 overflow-auto">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-slate-950">
                <tr>
                  <th className="border-b border-slate-800 px-3 py-2 text-xs font-semibold text-slate-300">Step</th>
                  <th className="border-b border-slate-800 px-3 py-2 text-xs font-semibold text-slate-300">OK</th>
                  <th className="border-b border-slate-800 px-3 py-2 text-xs font-semibold text-slate-300">Message</th>
                </tr>
              </thead>
              <tbody>
                {pipelineRes.steps.map((s) => (
                  <tr key={s.step} className="odd:bg-slate-900/20">
                    <td className="border-b border-slate-900 px-3 py-2 font-mono text-xs">{s.step}</td>
                    <td className="border-b border-slate-900 px-3 py-2">{s.ok ? "yes" : "no"}</td>
                    <td className="border-b border-slate-900 px-3 py-2 text-slate-200">{s.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}

