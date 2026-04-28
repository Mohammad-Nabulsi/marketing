import axios from "axios";
import type { DashboardResponse, PipelineSummary, UploadResponse } from "../types";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL,
  timeout: 60_000
});

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await api.post<UploadResponse>("/api/upload", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return res.data;
}

export async function runPipeline(datasetId: string): Promise<PipelineSummary> {
  const res = await api.post<PipelineSummary>(`/api/run-pipeline/${datasetId}`);
  return res.data;
}

export async function getKpis(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/kpis/${datasetId}`);
  return res.data;
}

export async function getContentPerformance(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/content-performance/${datasetId}`);
  return res.data;
}

export async function getClustering(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/clustering/${datasetId}`);
  return res.data;
}

export async function getRules(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/rules/${datasetId}`);
  return res.data;
}

export async function getTrends(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/trends/${datasetId}`);
  return res.data;
}

export async function getNetwork(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/network/${datasetId}`);
  return res.data;
}

export async function getRecommendations(datasetId: string): Promise<DashboardResponse> {
  const res = await api.get<DashboardResponse>(`/api/dashboard/recommendations/${datasetId}`);
  return res.data;
}

