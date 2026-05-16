"""
Temporal analysis adapter for the backend.

This module loads uploaded SME datasets, validates inputs, delegates to
existing business momentum, anomaly detection, and forecasting logic, and
returns frontend-friendly response dictionaries.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_SCRIPTS = PROJECT_ROOT / "notebooks" / "scripts"
GLOBAL_KPI_PATH = PROJECT_ROOT / "data" / "processed" / "kpi_dataset.csv"
GLOBAL_OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "outputs"
DYNAMIC_OUTPUT_DIR = Path(__file__).resolve().parents[0] / "storage" / "outputs"
DYNAMIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_identifier(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def _resolve_uploaded_path(uploaded_file_path: str) -> Path:
    path = Path(uploaded_file_path)
    if not path.is_absolute():
        candidates = [PROJECT_ROOT / uploaded_file_path, Path.cwd() / uploaded_file_path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    if path.exists():
        return path.resolve()
    raise FileNotFoundError(f"Uploaded file not found: {uploaded_file_path}")


def load_uploaded_dataset(uploaded_file_path: str) -> pd.DataFrame:
    """Load an uploaded CSV or JSON dataset from disk."""
    path = _resolve_uploaded_path(uploaded_file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(
        f"Uploaded dataset must be a .csv or .json file. Found: {path.suffix}"
    )


def validate_required_columns(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> None:
    """Validate that required columns are present in the uploaded dataset."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def build_frontend_response(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert response fields to frontend-safe values and return the payload."""
    cleaned = {}
    for key, value in response_payload.items():
        if isinstance(value, Path):
            cleaned[key] = str(value)
        elif isinstance(value, list):
            cleaned[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            cleaned[key] = value
    return cleaned


def _load_business_momentum_module():
    path = NOTEBOOKS_SCRIPTS / "04_business_momentum_weekly_trends.py"
    return _load_module("business_momentum_module", path)


def _load_anomaly_module():
    path = NOTEBOOKS_SCRIPTS / "05_anomaly_detection.py"
    return _load_module("anomaly_detection_module", path)


def _load_forecasting_module():
    path = NOTEBOOKS_SCRIPTS / "06_forecasting.py"
    return _load_module("forecasting_module", path)


def _safe_path_string(path) -> str:
    """Convert Path object to frontend-safe path string."""
    if path is None:
        return ""
    return str(Path(path).resolve()).replace("\\", "/")


def _safe_output_directory(label: str) -> Path:
    output_dir = DYNAMIC_OUTPUT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def _normalize_sector_query(sector: str) -> str:
    value = str(sector).strip().lower()

    sector_aliases = {
        "cafe": "cafes/restaurants",
        "cafes": "cafes/restaurants",
        "coffee": "cafes/restaurants",
        "restaurant": "cafes/restaurants",
        "restaurants": "cafes/restaurants",
        "مطعم": "cafes/restaurants",
        "مطاعم": "cafes/restaurants",
        "كافيه": "cafes/restaurants",
        "كافيهات": "cafes/restaurants",
        "كوفي": "cafes/restaurants",

        "fashion": "fashion",
        "clothes": "fashion",
        "clothing": "fashion",
        "ملابس": "fashion",
        "لبس": "fashion",
        "أزياء": "fashion",
        "ازياء": "fashion",

        "supermarket": "supermarkets",
        "supermarkets": "supermarkets",
        "market": "supermarkets",
        "grocery": "supermarkets",
        "سوبرماركت": "supermarkets",
        "بقالة": "supermarkets",
    }

    return sector_aliases.get(value, value)

def run_business_momentum_pipeline(uploaded_file_path: str) -> Dict[str, Any]:
    """Run business momentum comparison between uploaded business and sector benchmark."""
    momentum_module = _load_business_momentum_module()

    uploaded_df = load_uploaded_dataset(uploaded_file_path)
    required_columns = [
        "business_name",
        "sector",
        "post_date",
    ]
    validate_required_columns(uploaded_df, required_columns, "uploaded business dataset")

    if uploaded_df.empty:
        raise ValueError("Uploaded business dataset is empty.")

    business_name = str(uploaded_df["business_name"].dropna().astype(str).iloc[0])
    safe_business_name = _safe_identifier(business_name)
    output_dir = _safe_output_directory(f"business_momentum/{safe_business_name}")

    if not GLOBAL_KPI_PATH.exists():
        raise FileNotFoundError(f"Global sector dataset not found: {GLOBAL_KPI_PATH}")

    global_df = pd.read_csv(GLOBAL_KPI_PATH)
    result = momentum_module.run_ui_business_vs_sector_analysis(
        global_df,
        uploaded_df,
        output_dir,
        rolling_window=3,
        growth_threshold=0.10,
    )

    latest_summary = result["ui_business_momentum_summary"].iloc[-1]
    chart_path = result["chart_path"]
    csv_outputs = [
        str(output_dir / "sector_weekly_trends.csv"),
        str(output_dir / "uploaded_business_weekly_trends.csv"),
        str(output_dir / "business_vs_sector_momentum.csv"),
        str(output_dir / "ui_business_momentum_summary.csv"),
    ]

    return {
        "business_name": business_name,
        "sector": str(latest_summary["sector"]),
        "trend": str(latest_summary["trend"]),
        "comparison_label": str(latest_summary["comparison_label"]),
        "latest_business_engagement_rate": float(latest_summary["latest_business_engagement_rate"]),
        "latest_sector_engagement_rate": float(latest_summary["latest_sector_engagement_rate"]),
        "difference_from_sector": float(latest_summary["difference_from_sector"]),
        "message": str(latest_summary["recommendation_message"]),
        "chart_url": _safe_path_string(chart_path) if chart_path is not None else "",
        "csv_outputs": csv_outputs,
    }


def run_business_momentum_status(business_name: str) -> Dict[str, Any]:
    """Return latest business momentum and sector performance for a named business."""
    momentum_module = _load_business_momentum_module()

    comparison = getattr(momentum_module, "comparison", None)
    if comparison is None:
        raise ValueError("Business momentum comparison data is not available.")

    filtered = comparison[
        comparison["business_name"].astype(str).str.strip().str.lower()
        == business_name.strip().lower()
    ]
    if filtered.empty:
        raise ValueError(f"Business not found in global momentum data: {business_name}")

    latest = filtered.tail(1).iloc[-1]
    chart_path = GLOBAL_OUTPUT_DIR / "business_vs_sector_momentum.png"
    csv_outputs = []
    if chart_path.exists():
        csv_outputs.append(str(chart_path))
    saved_csv = GLOBAL_OUTPUT_DIR / "business_vs_sector_momentum.csv"
    if saved_csv.exists():
        csv_outputs.append(str(saved_csv))

    comparison_label = latest.get("performance_vs_sector")
    if comparison_label == "above_sector_average":
        comparison_label = "above sector average"
    elif comparison_label == "below_sector_average":
        comparison_label = "below sector average"
    else:
        comparison_label = str(comparison_label or "equal to sector average")

    message = (
        f"{business_name} has {latest['final_trend_class']} momentum and is {comparison_label}."
    )

    return {
        "business_name": str(latest["business_name"]),
        "sector": str(latest["sector"]),
        "trend": str(latest["final_trend_class"]),
        "comparison_label": comparison_label,
        "latest_business_engagement_rate": float(latest["latest_rolling_engagement_rate"]),
        "latest_sector_engagement_rate": float(latest.get("latest_sector_engagement_rate", np.nan)),
        "difference_from_sector": float(
            latest["latest_rolling_engagement_rate"]
            - latest.get("latest_sector_engagement_rate", 0)
        ),
        "message": message,
        "chart_url": str(chart_path) if chart_path.exists() else "",
        "csv_outputs": csv_outputs,
    }


def get_sector_momentum() -> List[Dict[str, Any]]:
    """Return latest momentum information for all sectors."""
    momentum_module = _load_business_momentum_module()
    sector_momentum = getattr(momentum_module, "sector_momentum", None)
    if sector_momentum is None:
        raise ValueError("Sector momentum data is not available.")

    return sector_momentum.to_dict(orient="records")


def _load_global_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path)


def get_top_positive_anomalies() -> List[Dict[str, Any]]:
    path = GLOBAL_OUTPUT_DIR / "top5_positive_anomalies_by_sector.csv"
    return _load_global_csv(path).to_dict(orient="records")


def get_top_negative_anomalies() -> List[Dict[str, Any]]:
    path = GLOBAL_OUTPUT_DIR / "top5_negative_anomalies_by_sector.csv"
    return _load_global_csv(path).to_dict(orient="records")


def get_anomaly_recommendations() -> List[Dict[str, Any]]:
    path = GLOBAL_OUTPUT_DIR / "anomaly_recommendations.csv"
    return _load_global_csv(path).to_dict(orient="records")


def get_sector_anomaly_summary(sector: str) -> List[Dict[str, Any]]:
    path = GLOBAL_OUTPUT_DIR / "sector_anomaly_pattern_summary.csv"
    rows = _load_global_csv(path)

    if "sector" not in rows.columns:
        raise ValueError("sector_anomaly_pattern_summary.csv does not contain a sector column.")

    normalized_query = _normalize_sector_query(sector)

    rows["_sector_normalized"] = (
        rows["sector"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered = rows[rows["_sector_normalized"] == normalized_query]

    if filtered.empty:
        available_sectors = sorted(rows["sector"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No anomaly sector summary found for sector: {sector}. "
            f"Available sectors: {available_sectors}"
        )

    return filtered.drop(columns=["_sector_normalized"]).to_dict(orient="records")


def run_anomaly_pipeline(uploaded_file_path: str) -> Dict[str, Any]:
    """Run anomaly detection and behavior recommendation for uploaded data."""
    anomaly_module = _load_anomaly_module()

    uploaded_df = load_uploaded_dataset(uploaded_file_path)
    validate_required_columns(uploaded_df, anomaly_module.required_columns, "uploaded anomaly dataset")

    uploaded_df["business_name"] = uploaded_df["business_name"].astype(str).str.strip()
    uploaded_df["sector"] = uploaded_df["sector"].astype(str).str.strip()

    if pd.api.types.is_numeric_dtype(uploaded_df["post_date"]):
        uploaded_df["post_date"] = pd.to_datetime(
            uploaded_df["post_date"],
            unit="ms",
            errors="coerce",
        )
    else:
        uploaded_df["post_date"] = pd.to_datetime(
            uploaded_df["post_date"],
            errors="coerce",
        )

    features = anomaly_module.features
    if not features:
        raise ValueError("Anomaly module does not expose required feature list.")

    business_name = (
        uploaded_df["business_name"].dropna().astype(str).iloc[0]
        if not uploaded_df["business_name"].dropna().empty
        else "Uploaded Dataset"
    )
    sectors = uploaded_df["sector"].dropna().unique().tolist()
    output_dir = _safe_output_directory(f"anomaly_detection/{_safe_identifier(business_name)}")

    all_anomalies: List[pd.DataFrame] = []
    all_sector_results: List[pd.DataFrame] = []
    top_positive_rows: List[pd.DataFrame] = []
    top_negative_rows: List[pd.DataFrame] = []
    pattern_rows: List[Dict[str, Any]] = []
    experiment_rows: List[Dict[str, Any]] = []

    for sector in sectors:
        sector_df = uploaded_df[uploaded_df["sector"] == sector].copy()
        if len(sector_df) < 10:
            continue

        X_raw = sector_df[features].fillna(0)
        X = StandardScaler().fit_transform(X_raw)

        method_outputs = {}

        for threshold in [2, 2.5, 3]:
            method_outputs[f"zscore_{threshold}"] = {
                "method": "zscore",
                "setting": f"threshold={threshold}",
                "y": anomaly_module.run_zscore(X, threshold),
            }

        for contamination in [0.03, 0.05, 0.10]:
            method_outputs[f"iforest_{contamination}"] = {
                "method": "isolation_forest",
                "setting": f"contamination={contamination}",
                "y": anomaly_module.run_iforest(X, contamination),
            }

        for n_neighbors in [10, 20, 35]:
            method_outputs[f"lof_{n_neighbors}"] = {
                "method": "lof",
                "setting": f"n_neighbors={n_neighbors}",
                "y": anomaly_module.run_lof(X, n_neighbors, contamination=0.05),
            }

        for method_key, info in method_outputs.items():
            anomaly_count, anomaly_ratio, balance_score, interpretability, final_score = anomaly_module.evaluate_method(
                info["y"],
                info["method"],
            )

            experiment_rows.append({
                "sector": sector,
                "method_key": method_key,
                "method": info["method"],
                "setting": info["setting"],
                "total_posts": len(sector_df),
                "anomaly_count": anomaly_count,
                "anomaly_ratio": anomaly_ratio,
                "balance_score": balance_score,
                "interpretability": interpretability,
                "final_score": final_score,
            })

        best_method_key = "iforest_0.05"
        fixed_method_info = method_outputs[best_method_key]
        best_y = fixed_method_info["y"]

        sector_result = anomaly_module.classify_anomaly_type(sector_df, best_y)
        sector_result["best_method"] = fixed_method_info["method"]
        sector_result["best_setting"] = fixed_method_info["setting"]
        sector_result["anomaly_detection_scope"] = "sector_based"

        all_sector_results.append(sector_result)

        sector_anomalies = sector_result[sector_result["anomaly_label"] == -1].copy()
        sector_anomalies = anomaly_module.add_anomaly_strength(sector_anomalies)

        all_anomalies.append(sector_anomalies)

        positive = (
            sector_anomalies[sector_anomalies["anomaly_type"] == "positive_anomaly"]
            .sort_values("engagement_rate", ascending=False)
            .head(5)
        )

        negative = (
            sector_anomalies[sector_anomalies["anomaly_type"] == "negative_anomaly"]
            .sort_values("engagement_rate", ascending=True)
            .head(5)
        )

        top_positive_rows.append(positive)
        top_negative_rows.append(negative)

        pattern_rows.append(
            anomaly_module.summarize_patterns(positive, sector, "positive_anomaly")
        )
        pattern_rows.append(
            anomaly_module.summarize_patterns(negative, sector, "negative_anomaly")
        )

    anomaly_experiments = pd.DataFrame(experiment_rows)
    all_anomalies_df = pd.concat(all_anomalies, ignore_index=True) if all_anomalies else pd.DataFrame()
    all_anomaly_results_df = pd.concat(all_sector_results, ignore_index=True) if all_sector_results else pd.DataFrame()
    top_positive_anomalies = pd.concat(top_positive_rows, ignore_index=True) if top_positive_rows else pd.DataFrame()
    top_negative_anomalies = pd.concat(top_negative_rows, ignore_index=True) if top_negative_rows else pd.DataFrame()
    sector_pattern_summary = pd.DataFrame(pattern_rows)

    all_anomalies_df.to_csv(output_dir / "sector_based_anomalies.csv", index=False)
    top_positive_anomalies.to_csv(output_dir / "top5_positive_anomalies_by_sector.csv", index=False)
    top_negative_anomalies.to_csv(output_dir / "top5_negative_anomalies_by_sector.csv", index=False)
    sector_pattern_summary.to_csv(output_dir / "sector_anomaly_pattern_summary.csv", index=False)
    anomaly_experiments.to_csv(output_dir / "sector_anomaly_experiments.csv", index=False)

    recommendation_outputs = anomaly_module.generate_anomaly_behavior_recommendations(
        all_anomaly_results_df,
        output_dir,
    )

    recommendations_df = recommendation_outputs["recommendations_df"]

    # Fallback recommendations:
    # If the normal recommendation engine returns empty results, still provide useful
    # business guidance based on strongest and weakest posts.
    if recommendations_df.empty:
        fallback_rows = []

        if not top_positive_anomalies.empty:
            common_post_type = (
                top_positive_anomalies["post_type"].mode().iloc[0]
                if "post_type" in top_positive_anomalies.columns
                and not top_positive_anomalies["post_type"].mode().empty
                else "content"
            )

            avg_caption_length = (
                top_positive_anomalies["caption_length"].mean()
                if "caption_length" in top_positive_anomalies.columns
                else np.nan
            )

            avg_hashtags = (
                top_positive_anomalies["hashtags_count"].mean()
                if "hashtags_count" in top_positive_anomalies.columns
                else np.nan
            )

            avg_emojis = (
                top_positive_anomalies["emoji_count"].mean()
                if "emoji_count" in top_positive_anomalies.columns
                else np.nan
            )

            fallback_rows.append({
                "recommendation_type": "continue_doing",
                "behavior_feature": "post_type",
                "friendly_action": f"continue using {common_post_type} posts",
                "direction": "repeat_successful_behavior",
                "relative_difference_percent": np.nan,
                "recommendation_message": (
                    f"Continue doing this: your strongest anomalous posts are mostly "
                    f"{common_post_type} posts. Keep using this format because it is "
                    "repeatedly linked with high engagement."
                ),
            })

            fallback_rows.append({
                "recommendation_type": "continue_doing",
                "behavior_feature": "content_style",
                "friendly_action": "keep a similar winning content style",
                "direction": "repeat_successful_behavior",
                "relative_difference_percent": np.nan,
                "recommendation_message": (
                    f"Your strongest posts have an average caption length around "
                    f"{avg_caption_length:.0f}, average hashtags around {avg_hashtags:.0f}, "
                    f"and average emojis around {avg_emojis:.0f}. Use this as a safe "
                    "creative benchmark for future posts."
                ),
            })

        weakest_posts = uploaded_df.copy()
        if "engagement_rate" in weakest_posts.columns:
            weakest_posts["engagement_rate"] = pd.to_numeric(
                weakest_posts["engagement_rate"],
                errors="coerce",
            )
            weakest_posts = weakest_posts.dropna(subset=["engagement_rate"])
            weakest_posts = weakest_posts.sort_values("engagement_rate", ascending=True).head(5)
        else:
            weakest_posts = pd.DataFrame()

        if not weakest_posts.empty:
            weak_post_type = (
                weakest_posts["post_type"].mode().iloc[0]
                if "post_type" in weakest_posts.columns
                and not weakest_posts["post_type"].mode().empty
                else "content"
            )

            weak_caption_length = (
                weakest_posts["caption_length"].mean()
                if "caption_length" in weakest_posts.columns
                else np.nan
            )

            fallback_rows.append({
                "recommendation_type": "be_careful",
                "behavior_feature": "weakest_posts",
                "friendly_action": f"be careful with {weak_post_type} posts",
                "direction": "avoid_overusing_weak_behavior",
                "relative_difference_percent": np.nan,
                "recommendation_message": (
                    f"Be careful: your lowest-performing posts are often {weak_post_type} "
                    f"posts with average caption length around {weak_caption_length:.0f}. "
                    "Do not rely too much on this style unless you test and improve it."
                ),
            })

        if fallback_rows:
            recommendations_df = pd.DataFrame(fallback_rows)

            # Save fallback recommendations to the same recommendations CSV file
            # so the frontend and downloadable outputs are never empty.
            recommendations_df.to_csv(
                output_dir / "anomaly_recommendations.csv",
                index=False,
            )

    csv_outputs = [
        str(output_dir / "sector_based_anomalies.csv"),
        str(output_dir / "top5_positive_anomalies_by_sector.csv"),
        str(output_dir / "top5_negative_anomalies_by_sector.csv"),
        str(output_dir / "sector_anomaly_pattern_summary.csv"),
        str(output_dir / "sector_anomaly_experiments.csv"),
        str(recommendation_outputs.get("profile_summary_path")),
        str(output_dir / "anomaly_recommendations.csv"),
    ]

    if recommendation_outputs.get("chart_path"):
        csv_outputs.append(str(recommendation_outputs["chart_path"]))

    message = "Anomaly detection completed successfully."
    if all_anomaly_results_df.empty:
        message = (
            "Anomaly detection finished, but no sector anomaly results were produced "
            "from the uploaded dataset."
        )
    elif top_negative_anomalies.empty:
        message = (
            "Anomaly detection completed successfully. No strong negative anomalies "
            "were found, so recommendations focus on continuing successful behaviors "
            "and being careful with the lowest-performing posts."
        )

    return {
        "business_name": business_name,
        "sector": sectors[0] if len(sectors) == 1 else "mixed",
        "message": message,
        "top_positive_anomalies": top_positive_anomalies.to_dict(orient="records"),
        "top_negative_anomalies": top_negative_anomalies.to_dict(orient="records"),
        "recommendations": recommendations_df.to_dict(orient="records"),
        "chart_url": str(recommendation_outputs["chart_path"]) if recommendation_outputs.get("chart_path") else None,
        "csv_outputs": csv_outputs,
    }

def run_forecasting_pipeline(uploaded_file_path: str) -> Dict[str, Any]:
    """Run forecasting analysis on an uploaded dataset."""
    forecasting_module = _load_forecasting_module()
    business_name = _safe_identifier(uploaded_file_path)
    output_dir = _safe_output_directory(f"forecast/{business_name}")

    uploaded_df = load_uploaded_dataset(uploaded_file_path)
    result = forecasting_module.run_forecasting_analysis(
    uploaded_df,
    output_dir,
    periods=8,
)


    summary_df = result["summary_df"].iloc[0]
    forecast_df = result["forecast_df"].copy()
    future_forecast = forecast_df[forecast_df["forecast_type"] == "future"].copy()
    future_forecast["ds"] = pd.to_datetime(future_forecast["ds"]).dt.strftime("%Y-%m-%d")

    csv_outputs = [
        str(result["forecast_path"]),
        str(result["metrics_path"]),
        str(result["summary_path"]),
    ]

    return {
        "business_name": str(summary_df["business_name"]),
        "best_model": str(summary_df["best_model"]),
        "forecast_horizon_weeks": int(summary_df["forecast_horizon_weeks"]),
        "best_MAE": float(summary_df["best_MAE"]),
        "best_RMSE": float(summary_df["best_RMSE"]),
        "message": f"Forecast completed using {summary_df['best_model']} for {summary_df['business_name']}.",
        "future_forecast": future_forecast.to_dict(orient="records"),
        "chart_url": str(result["chart_path"]),
        "csv_outputs": csv_outputs,
    }


def get_saved_forecast(business_name: str) -> Dict[str, Any]:
    """Return a previously saved forecast summary for a business."""
    summary_files = list(DYNAMIC_OUTPUT_DIR.rglob("*_summary.csv"))
    summary_files += list(GLOBAL_OUTPUT_DIR.rglob("*_summary.csv"))

    for path in summary_files:
        summary = pd.read_csv(path)
        matched = summary[summary["business_name"].astype(str).str.strip().str.lower() == business_name.strip().lower()]
        if not matched.empty:
            row = matched.iloc[0]
            prefix = path.stem.replace("_summary", "")
            forecast_path = path.with_name(f"{prefix}.csv")
            if not forecast_path.exists():
                raise FileNotFoundError(f"Forecast CSV is missing for saved forecast: {forecast_path}")
            forecast_df = pd.read_csv(forecast_path)
            future_forecast = forecast_df[forecast_df["forecast_type"] == "future"].copy()
            future_forecast["ds"] = pd.to_datetime(future_forecast["ds"]).dt.strftime("%Y-%m-%d")
            return {
                "business_name": str(row["business_name"]),
                "best_model": str(row["best_model"]),
                "forecast_horizon_weeks": int(row["forecast_horizon_weeks"]),
                "best_MAE": float(row["best_MAE"]),
                "best_RMSE": float(row["best_RMSE"]),
                "message": f"Saved forecast found for {row['business_name']}",
                "future_forecast": future_forecast.to_dict(orient="records"),
                "chart_url": str(path.with_name(f"{prefix}.png")) if path.with_name(f"{prefix}.png").exists() else "",
                "csv_outputs": [str(path), str(forecast_path)],
            }

    raise FileNotFoundError(f"No saved forecast found for business: {business_name}")
