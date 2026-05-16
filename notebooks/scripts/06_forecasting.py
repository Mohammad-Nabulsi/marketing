# 06 Weekly Forecasting
# Social media engagement dataset, weekly forecasting with Prophet + simple baselines.

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

sys.path.insert(0, str(NOTEBOOKS_DIR))

from utils.utils import ensure_project_dirs
from utils.evaluation import regression_metrics, rank_models

ensure_project_dirs()

DATASET_PATH = PROJECT_ROOT / "data" / "social_media_engagement_dataset.csv"

OUTPUTS_DIR = NOTEBOOKS_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Forecasting Dataset


def load_forecasting_input(data=None):
    if isinstance(data, pd.DataFrame):
        return data.copy()

    if data is None:
        return pd.read_csv(DATASET_PATH)

    return pd.read_csv(data)


def _normalize_column_name(column_name):
    return (
        str(column_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _find_first_available_column(frame, candidates):
    normalized_lookup = {
        _normalize_column_name(col): col
        for col in frame.columns
    }

    for candidate in candidates:
        normalized_candidate = _normalize_column_name(candidate)
        if normalized_candidate in normalized_lookup:
            return normalized_lookup[normalized_candidate]

    return None


def infer_forecasting_columns(frame):
    date_col = _find_first_available_column(
        frame,
        [
            "post_date",
            "timestamp",
            "date",
            "created_at",
            "published_at",
            "datetime",
            "time",
        ],
    )

    target_col = _find_first_available_column(
        frame,
        [
            "engagement_rate",
            "engagement rate",
            "Engagement_Rate",
            "avg_engagement_rate",
            "views",
            "views_count",
            "likes",
            "likes_count",
            "comments",
            "comments_count",
        ],
    )

    entity_col = _find_first_available_column(
        frame,
        [
            "business_name",
            "business",
            "brand",
            "account",
            "page",
            "platform",
            "Platform",
        ],
    )

    sector_col = _find_first_available_column(
        frame,
        [
            "sector",
            "category",
            "Category",
            "industry",
        ],
    )

    return {
        "date_col": date_col,
        "target_col": target_col,
        "entity_col": entity_col,
        "sector_col": sector_col,
    }


def prepare_forecasting_dataframe(
    data,
    date_col=None,
    target_col=None,
    entity_col=None,
    sector_col=None,
):
    frame = data.copy()

    inferred_columns = infer_forecasting_columns(frame)
    date_col = date_col or inferred_columns["date_col"]
    target_col = target_col or inferred_columns["target_col"]
    entity_col = entity_col or inferred_columns["entity_col"]
    sector_col = sector_col or inferred_columns["sector_col"]

    if date_col is None or target_col is None:
        raise ValueError(
            "Could not infer required forecasting columns. "
            "Please provide date_col and target_col."
        )

    rename_map = {
        date_col: "post_date",
        target_col: "target_value",
    }

    if entity_col is not None:
        rename_map[entity_col] = "business_name"

    if sector_col is not None:
        rename_map[sector_col] = "sector"

    frame = frame.rename(columns=rename_map)

    if "business_name" not in frame.columns:
        frame["business_name"] = "Uploaded Dataset"

    if "sector" not in frame.columns:
        frame["sector"] = "General"

    required_columns = [
        "business_name",
        "sector",
        "post_date",
        "target_value",
    ]

    missing_columns = [col for col in required_columns if col not in frame.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    frame["business_name"] = frame["business_name"].astype(str).str.strip()
    frame["sector"] = frame["sector"].astype(str).str.strip()

    if pd.api.types.is_numeric_dtype(frame["post_date"]):
        frame["post_date"] = pd.to_datetime(
            frame["post_date"],
            unit="ms",
            errors="coerce"
        )
    else:
        frame["post_date"] = pd.to_datetime(
            frame["post_date"],
            errors="coerce"
        )

    frame["target_value"] = pd.to_numeric(
        frame["target_value"],
        errors="coerce"
    )

    frame = frame.dropna(
        subset=[
            "business_name",
            "sector",
            "post_date",
            "target_value",
        ]
    )

    upper_limit = frame["target_value"].quantile(0.99)

    frame["target_value_clipped"] = np.clip(
        frame["target_value"],
        None,
        upper_limit
    )

    frame["forecast_target_name"] = target_col
    frame.attrs["date_col"] = date_col
    frame.attrs["target_col"] = target_col
    frame.attrs["entity_col"] = entity_col
    frame.attrs["sector_col"] = sector_col

    return frame

# Build Weekly Time Series


def build_weekly_ts(frame):
    ts = (
        frame.set_index("post_date")["target_value_clipped"]
        .resample("W-MON")
        .mean()
        .dropna()
        .reset_index()
        .rename(
            columns={
                "post_date": "ds",
                "target_value_clipped": "y",
            }
        )
    )

    ts["ds"] = pd.to_datetime(ts["ds"])
    return ts

# Forecast

from prophet import Prophet


def make_future_dates(last_date, periods):
    return pd.date_range(
        start=last_date + pd.offsets.Week(weekday=0),
        periods=periods,
        freq="W-MON",
    )


def build_constant_forecast(test, future_dates, yhat_value):
    test_pred = test[["ds", "y"]].copy()
    test_pred["yhat"] = yhat_value
    test_pred["yhat_lower"] = np.nan
    test_pred["yhat_upper"] = np.nan
    test_pred["forecast_type"] = "test"

    future_pred = pd.DataFrame({
        "ds": future_dates,
        "y": np.nan,
        "yhat": yhat_value,
        "yhat_lower": np.nan,
        "yhat_upper": np.nan,
        "forecast_type": "future",
    })

    return pd.concat([test_pred, future_pred], ignore_index=True)


def run_forecast(ts, business_name, target_name, periods=8):
    ts = ts.sort_values("ds").reset_index(drop=True)

    split = int(len(ts) * 0.8)

    train = ts.iloc[:split]
    test = ts.iloc[split:]

    if len(test) < 2:
        raise ValueError(
            f"Not enough test weeks. Test weeks: {len(test)}"
        )

    experiment_rows = []
    prediction_store = {}

    future_dates = make_future_dates(ts["ds"].max(), periods)

    baseline_values = {
        "naive_last_value": train["y"].iloc[-1],
        "moving_average_4": train["y"].tail(4).mean(),
        "moving_average_8": train["y"].tail(8).mean(),
    }

    for model_key, yhat_value in baseline_values.items():
        pred = pd.DataFrame({
            "ds": test["ds"],
            "yhat": yhat_value,
        })

        metrics = regression_metrics(test["y"], pred["yhat"])

        experiment_rows.append({
            "business_name": business_name,
            "model": model_key,
            "aggregation": "weekly",
            "changepoint_prior_scale": np.nan,
            "seasonality_mode": "not_applicable",
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            **metrics,
        })

        prediction_store[model_key] = {
            "model_type": "baseline",
            "test_prediction": pred,
            "yhat_value": yhat_value,
        }

    cp_vals = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
    seasonality_modes = ["additive", "multiplicative"]
    yearly_options = [False, True]

    for cp in cp_vals:
        for seasonality_mode in seasonality_modes:
            for yearly in yearly_options:
                model_key = f"prophet_cp_{cp}_{seasonality_mode}_yearly_{yearly}"

                model = Prophet(
                    changepoint_prior_scale=cp,
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                )

                model.fit(train)

                future = model.make_future_dataframe(
                    periods=len(test),
                    freq="W-MON"
                )

                pred = model.predict(future)[["ds", "yhat"]].tail(len(test))
                pred["yhat"] = pred["yhat"].clip(lower=0)

                metrics = regression_metrics(test["y"], pred["yhat"])

                experiment_rows.append({
                    "business_name": business_name,
                    "model": "prophet",
                    "model_key": model_key,
                    "aggregation": "weekly",
                    "changepoint_prior_scale": cp,
                    "seasonality_mode": seasonality_mode,
                    "yearly_seasonality": yearly,
                    "weekly_seasonality": False,
                    **metrics,
                })

                prediction_store[model_key] = {
                    "model_type": "prophet",
                    "test_prediction": pred,
                    "changepoint_prior_scale": cp,
                    "seasonality_mode": seasonality_mode,
                    "yearly_seasonality": yearly,
                }

    exp_df = pd.DataFrame(experiment_rows)

    exp_df["model_key"] = exp_df["model_key"].fillna(exp_df["model"])

    ranked = rank_models(
        exp_df,
        lower_is_better_cols=["MAE", "RMSE", "MAPE"]
    )

    best = ranked.iloc[0]
    best_key = best["model_key"]
    best_prediction = prediction_store[best_key]

    if best_prediction["model_type"] == "baseline":
        forecast = build_constant_forecast(
            test,
            future_dates,
            best_prediction["yhat_value"],
        )

    else:
        test_forecast = test.merge(
            best_prediction["test_prediction"],
            on="ds",
            how="left",
        )
        test_forecast["forecast_type"] = "test"
        test_forecast["yhat_lower"] = np.nan
        test_forecast["yhat_upper"] = np.nan

        final_model = Prophet(
            changepoint_prior_scale=float(best_prediction["changepoint_prior_scale"]),
            seasonality_mode=best_prediction["seasonality_mode"],
            yearly_seasonality=bool(best_prediction["yearly_seasonality"]),
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        final_model.fit(ts)

        future = final_model.make_future_dataframe(
            periods=periods,
            freq="W-MON"
        )

        full_forecast = final_model.predict(future)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ]

        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            full_forecast[col] = full_forecast[col].clip(lower=0)

        future_forecast = full_forecast[
            full_forecast["ds"] > ts["ds"].max()
        ].copy()

        future_forecast["y"] = np.nan
        future_forecast["forecast_type"] = "future"

        forecast = pd.concat(
            [
                test_forecast[
                    ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "forecast_type"]
                ],
                future_forecast[
                    ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "forecast_type"]
                ],
            ],
            ignore_index=True,
        )

    forecast["business_name"] = business_name
    forecast["model"] = best["model"]
    forecast["model_key"] = best_key
    forecast["aggregation"] = "weekly"
    forecast["target"] = target_name
    forecast["target_transform"] = "clipped_at_99th_percentile"
    forecast["changepoint_prior_scale"] = best["changepoint_prior_scale"]
    forecast["seasonality_mode"] = best["seasonality_mode"]
    forecast["yearly_seasonality"] = best["yearly_seasonality"]
    forecast["weekly_seasonality"] = best["weekly_seasonality"]

    return forecast, ranked


def plot_forecast(forecast, title, ylabel, chart_path, show_plot=False):
    test_part = forecast[forecast["forecast_type"] == "test"]
    future_part = forecast[forecast["forecast_type"] == "future"]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        test_part["ds"],
        test_part["y"],
        marker="o",
        label="Actual",
    )

    ax.plot(
        test_part["ds"],
        test_part["yhat"],
        marker="o",
        label="Predicted",
    )

    ax.plot(
        future_part["ds"],
        future_part["yhat"],
        marker="o",
        linestyle="--",
        label="Future Forecast",
    )

    if future_part["yhat_lower"].notna().any() and future_part["yhat_upper"].notna().any():
        ax.fill_between(
            future_part["ds"],
            future_part["yhat_lower"],
            future_part["yhat_upper"],
            alpha=0.2,
        )

    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    fig.savefig(chart_path, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def run_forecasting_analysis(
    uploaded_dataset,
    output_dir,
    periods=8,
    date_col=None,
    target_col=None,
    entity_col=None,
    sector_col=None,
    output_prefix="uploaded_weekly_forecast",
    show_plot=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_forecasting_input(uploaded_dataset)
    prepared_df = prepare_forecasting_dataframe(
        raw_df,
        date_col=date_col,
        target_col=target_col,
        entity_col=entity_col,
        sector_col=sector_col,
    )

    business_names = prepared_df["business_name"].dropna().unique()
    if len(business_names) == 1:
        business_name = business_names[0]
    else:
        entity_name = prepared_df.attrs.get("entity_col") or "Dataset"
        entity_name = str(entity_name).replace("_", " ").title()
        if not entity_name.endswith("s"):
            entity_name = f"{entity_name}s"
        business_name = f"All {entity_name}"

    target_name = prepared_df["forecast_target_name"].iloc[0]
    weekly_ts = build_weekly_ts(prepared_df)

    if len(weekly_ts) < 12:
        raise ValueError(
            f"Not enough weekly data for forecasting. "
            f"Available weekly points: {len(weekly_ts)}"
        )

    forecast, forecast_metrics = run_forecast(
        weekly_ts,
        business_name,
        target_name,
        periods=periods,
    )

    forecast_path = output_dir / f"{output_prefix}.csv"
    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    chart_path = output_dir / f"{output_prefix}.png"
    summary_path = output_dir / f"{output_prefix}_summary.csv"

    best_model = forecast_metrics.iloc[0]
    summary_df = pd.DataFrame(
        [
            {
                "business_name": business_name,
                "target_column": target_name,
                "date_start": prepared_df["post_date"].min(),
                "date_end": prepared_df["post_date"].max(),
                "input_rows": len(prepared_df),
                "weekly_points": len(weekly_ts),
                "forecast_horizon_weeks": periods,
                "best_model": best_model["model"],
                "best_model_key": best_model["model_key"],
                "best_MAE": best_model["MAE"],
                "best_RMSE": best_model["RMSE"],
                "best_MAPE": best_model["MAPE"],
            }
        ]
    )

    forecast.to_csv(forecast_path, index=False)
    forecast_metrics.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_forecast(
        forecast,
        f"{business_name} Weekly {target_name} Forecast",
        target_name,
        chart_path,
        show_plot=show_plot,
    )

    return {
        "prepared_df": prepared_df,
        "weekly_ts": weekly_ts,
        "forecast_df": forecast,
        "forecast_metrics_df": forecast_metrics,
        "summary_df": summary_df,
        "forecast_path": forecast_path,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "chart_path": chart_path,
    }


if __name__ == "__main__":
    result = run_forecasting_analysis(
        DATASET_PATH,
        OUTPUTS_DIR,
        periods=8,
        output_prefix="social_media_weekly_prophet_forecast",
        show_plot=True,
    )

    summary = result["summary_df"].iloc[0]

    print("Social Media Weekly Forecasting completed successfully.")
    print()
    print("Business:")
    print(summary["business_name"])
    print()
    print("Input file:")
    print(DATASET_PATH)
    print()
    print("Saved outputs to:")
    print(OUTPUTS_DIR)
    print()
    print("Generated files:")
    print("- social_media_weekly_prophet_forecast.csv")
    print("- social_media_weekly_prophet_forecast_metrics.csv")
    print("- social_media_weekly_prophet_forecast_summary.csv")
    print("- social_media_weekly_prophet_forecast.png")
    print()
    print("Forecast target:")
    print(f"- {summary['target_column']}")
    print()
    print("Forecast horizon:")
    print(f"- next {summary['forecast_horizon_weeks']} weeks")
