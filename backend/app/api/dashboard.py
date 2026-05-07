from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.schemas import DashboardResponse
from app.services.similar_business_recommender import generate_similar_business_recommendations
from app.utils.file_utils import read_json, safe_read_csv


router = APIRouter(prefix="/dashboard")


def _outputs_dir(dataset_id: str) -> Path:
    return settings.storage_path() / "outputs" / dataset_id


def _read_csv_records(path: Path, max_rows: int = 5000):
    df = safe_read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_dict(orient="records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@router.get("/kpis/{dataset_id}", response_model=DashboardResponse)
def get_kpis(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    kpis_path = out / "kpis.csv"
    eda_path = out / "eda_summary.json"
    if not kpis_path.exists():
        raise HTTPException(status_code=404, detail="KPIs not found. Run pipeline first.")
    data = {
        "kpis_preview": _read_csv_records(kpis_path, max_rows=200),
        "eda_summary": read_json(eda_path) if eda_path.exists() else {},
    }
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/content-performance/{dataset_id}", response_model=DashboardResponse)
def content_performance(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    eda_path = out / "eda_summary.json"
    if not eda_path.exists():
        raise HTTPException(status_code=404, detail="EDA summary not found. Run pipeline first.")
    return DashboardResponse(dataset_id=dataset_id, data=read_json(eda_path))


@router.get("/clustering/{dataset_id}", response_model=DashboardResponse)
def clustering(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    post_clusters = out / "post_clusters.csv"
    biz_clusters = out / "business_clusters.csv"
    post_pca = out / "post_pca.csv"
    biz_pca = out / "business_pca.csv"
    if not post_clusters.exists() or not biz_clusters.exists():
        raise HTTPException(status_code=404, detail="Clustering outputs not found. Run pipeline first.")
    data = {
        "post_clusters": _read_csv_records(post_clusters, max_rows=2000),
        "business_clusters": _read_csv_records(biz_clusters, max_rows=2000),
        "post_pca": _read_csv_records(post_pca, max_rows=3000) if post_pca.exists() else [],
        "business_pca": _read_csv_records(biz_pca, max_rows=3000) if biz_pca.exists() else [],
    }
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/rules/{dataset_id}", response_model=DashboardResponse)
def rules(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    rules_path = out / "association_rules.csv"
    bv_rules_path = out / "business_value_rules.csv"
    if not rules_path.exists():
        raise HTTPException(status_code=404, detail="Association rules not found. Run pipeline first.")
    data = {
        "association_rules": _read_csv_records(rules_path, max_rows=2000),
        "business_value_rules": _read_csv_records(bv_rules_path, max_rows=2000) if bv_rules_path.exists() else [],
    }
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/trends/{dataset_id}", response_model=DashboardResponse)
def trends(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    weekly = out / "weekly_trends.csv"
    momentum = out / "business_momentum.csv"
    forecast = out / "forecast.csv"
    anomalies = out / "anomalies.csv"
    if not weekly.exists():
        raise HTTPException(status_code=404, detail="Trend outputs not found. Run pipeline first.")
    data = {
        "weekly_trends": _read_csv_records(weekly, max_rows=2000),
        "business_momentum": _read_csv_records(momentum, max_rows=2000) if momentum.exists() else [],
        "forecast": _read_csv_records(forecast, max_rows=2000) if forecast.exists() else [],
        "anomalies": _read_csv_records(anomalies, max_rows=2000) if anomalies.exists() else [],
    }
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/network/{dataset_id}", response_model=DashboardResponse)
def network(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    nodes = out / "network_nodes.csv"
    edges = out / "network_edges.csv"
    summary = out / "network_summary.json"
    if not nodes.exists() or not edges.exists():
        raise HTTPException(status_code=404, detail="Network outputs not found. Run pipeline first.")
    data = {
        "nodes": _read_csv_records(nodes, max_rows=5000),
        "edges": _read_csv_records(edges, max_rows=5000),
        "summary": read_json(summary) if summary.exists() else {},
    }
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/recommendations/{dataset_id}", response_model=DashboardResponse)
def recommendations(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    recs = out / "recommendations.csv"
    if not recs.exists():
        raise HTTPException(status_code=404, detail="Recommendations not found. Run pipeline first.")
    data = {"recommendations": _read_csv_records(recs, max_rows=5000)}
    return DashboardResponse(dataset_id=dataset_id, data=data)


@router.get("/similar-businesses/{dataset_id}/businesses", response_model=DashboardResponse)
def similar_business_options(dataset_id: str) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    kpis_path = out / "kpis.csv"

    if not kpis_path.exists():
        raise HTTPException(status_code=404, detail="KPIs not found. Run pipeline first.")

    df = safe_read_csv(kpis_path)
    options = (
        df[["business_name", "sector"]]
        .dropna()
        .drop_duplicates()
        .sort_values(by=["sector", "business_name"])
        .to_dict(orient="records")
    )

    return DashboardResponse(
        dataset_id=dataset_id,
        data={"businesses": _json_safe(options)},
    )


@router.get("/similar-businesses/{dataset_id}", response_model=DashboardResponse)
def similar_business_recommendations(
    dataset_id: str,
    business_name: str = Query(..., min_length=1),
    sector: str | None = Query(default=None),
    top_n: int = Query(default=5, ge=1, le=10),
) -> DashboardResponse:
    out = _outputs_dir(dataset_id)
    kpis_path = out / "kpis.csv"

    if not kpis_path.exists():
        raise HTTPException(status_code=404, detail="KPIs not found. Run pipeline first.")

    df = safe_read_csv(kpis_path)

    try:
        recommendations_df, metadata = generate_similar_business_recommendations(
            posts_df=df,
            business_name=business_name,
            sector=sector,
            top_n=top_n,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Similar business recommender failed: {error}")

    data = {
        "recommendations": recommendations_df.to_dict(orient="records"),
        "metadata": metadata,
    }

    return DashboardResponse(
        dataset_id=dataset_id,
        data=_json_safe(data),
    )

