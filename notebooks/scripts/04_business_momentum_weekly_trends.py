# 04 Business Momentum Weekly Trends
# Sector-level trends + Business-level momentum analysis

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Outputs folder
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(NOTEBOOKS_DIR))

from utils.utils import ensure_project_dirs
from utils.visualization import set_plot_style

set_plot_style()
ensure_project_dirs()

KPI_PATH = PROJECT_ROOT / "data" / "processed" / "kpi_dataset.csv"


# Load KPI Dataset


# df = pd.read_json(KPI_PATH)
df = pd.read_csv(KPI_PATH)


df["post_date"] = pd.to_datetime(df["post_date"], unit="ms", errors="coerce")

required_columns = [
    "business_name",
    "sector",
    "post_date",
    "engagement",
    "engagement_rate",
    "week",
    "likes_count",
    "comments_count",
    "views_count",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        "This script expects the dataset exported from 00_kpi_engineering. "
        f"Missing columns: {missing_columns}"
    )

df = df.dropna(subset=["business_name", "sector", "week"])

# 3 Helper Function


def classify_trend(growth, threshold=0.10):
    if growth > threshold:
        return "improving"
    elif growth < -threshold:
        return "declining"
    else:
        return "stable"


def _ensure_post_date_datetime(dataframe):
    dataframe = dataframe.copy()

    if pd.api.types.is_numeric_dtype(dataframe["post_date"]):
        dataframe["post_date"] = pd.to_datetime(
            dataframe["post_date"],
            unit="ms",
            errors="coerce",
        )
    else:
        dataframe["post_date"] = pd.to_datetime(
            dataframe["post_date"],
            errors="coerce",
        )

    return dataframe


def _ensure_engagement_rate(dataframe):
    dataframe = dataframe.copy()

    if "engagement_rate" not in dataframe.columns:
        dataframe["engagement"] = (
            dataframe["likes_count"].fillna(0)
            + 2 * dataframe["comments_count"].fillna(0)
            + 0.1 * dataframe["views_count"].fillna(0)
        )
        followers = dataframe["followers_count"].replace(0, np.nan)
        dataframe["engagement_rate"] = dataframe["engagement"] / followers

    if "engagement" not in dataframe.columns:
        dataframe["engagement"] = dataframe["engagement_rate"]

    dataframe["engagement_rate"] = (
        pd.to_numeric(dataframe["engagement_rate"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    return dataframe


def _prepare_ui_dataset(dataframe):
    dataframe = _ensure_post_date_datetime(dataframe)
    dataframe = _ensure_engagement_rate(dataframe)
    dataframe = dataframe.dropna(subset=["post_date", "engagement_rate"]).copy()
    dataframe["week"] = dataframe["post_date"].dt.to_period("W").dt.start_time

    return dataframe


def _safe_pct_change(series):
    return (
        series.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def _classify_weekly_trend(weekly_growth, threshold=0.10):
    clean_growth = weekly_growth.dropna()

    if clean_growth.empty:
        return "stable"

    trend_states = clean_growth.apply(lambda x: classify_trend(x, threshold))
    has_improving = (trend_states == "improving").any()
    has_declining = (trend_states == "declining").any()

    if has_improving and has_declining:
        return "inconsistent"

    latest_growth = clean_growth.iloc[-1]
    return classify_trend(latest_growth, threshold)


def _build_recommendation_message(
    business_name,
    sector,
    trend,
    comparison_label,
    latest_difference,
):
    if comparison_label == "above sector average" and trend == "improving":
        return (
            f"{business_name} is outperforming the {sector} sector and momentum is improving. "
            "Keep scaling the strongest content themes."
        )

    if comparison_label == "above sector average":
        return (
            f"{business_name} is above the {sector} sector average. "
            "Protect the current content rhythm and test small improvements to regain growth."
        )

    if comparison_label == "below sector average" and trend == "declining":
        return (
            f"{business_name} is below the {sector} sector average and engagement is declining. "
            "Prioritize a content refresh and review posting cadence, hooks, and audience fit."
        )

    if comparison_label == "below sector average":
        return (
            f"{business_name} is below the {sector} sector average by {latest_difference:.4f}. "
            "Focus on formats and topics that are already working for stronger sector performers."
        )

    return (
        f"{business_name} is currently equal to the {sector} sector average. "
        "Use the next posts to test creative variations that can move performance above benchmark."
    )


def run_ui_business_vs_sector_analysis(
    global_dataset_df,
    uploaded_business_dataset_df,
    output_dir,
    rolling_window=3,
    growth_threshold=0.10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_df = _prepare_ui_dataset(global_dataset_df.copy())
    uploaded_df = _prepare_ui_dataset(uploaded_business_dataset_df.copy())

    business_name = uploaded_df["business_name"].dropna().astype(str).iloc[0]
    sector = uploaded_df["sector"].dropna().astype(str).iloc[0]

    global_df = global_df.dropna(subset=["sector", "week"]).copy()
    sector_weekly_all = (
        global_df.groupby(["sector", "week"], as_index=False)
        .agg(
            sector_engagement_rate=("engagement_rate", "mean"),
            sector_total_posts=("engagement_rate", "size"),
        )
        .sort_values(["sector", "week"])
    )

    sector_weekly_all["sector_weekly_growth"] = (
        sector_weekly_all.groupby("sector")["sector_engagement_rate"]
        .transform(_safe_pct_change)
    )

    sector_weekly = sector_weekly_all[
        sector_weekly_all["sector"].astype(str) == sector
    ].copy()

    if sector_weekly.empty:
        raise ValueError(f"No sector weekly data found for sector: {sector}")

    uploaded_business_weekly = (
        uploaded_df.groupby(["business_name", "week"], as_index=False)
        .agg(
            business_engagement_rate=("engagement_rate", "mean"),
            business_total_posts=("engagement_rate", "size"),
        )
        .sort_values(["business_name", "week"])
    )

    if uploaded_business_weekly.empty:
        raise ValueError("uploaded business dataset has no weekly engagement data.")

    uploaded_business_weekly["rolling_engagement_rate"] = (
        uploaded_business_weekly.groupby("business_name")["business_engagement_rate"]
        .transform(lambda s: s.rolling(rolling_window, min_periods=1).mean())
    )
    uploaded_business_weekly["weekly_growth"] = (
        uploaded_business_weekly.groupby("business_name")["rolling_engagement_rate"]
        .transform(_safe_pct_change)
    )

    comparison = uploaded_business_weekly.merge(
        sector_weekly[["sector", "week", "sector_engagement_rate", "sector_total_posts"]],
        on="week",
        how="left",
    )
    comparison["sector"] = comparison["sector"].fillna(sector)
    comparison["difference_from_sector"] = (
        comparison["business_engagement_rate"]
        - comparison["sector_engagement_rate"]
    )

    comparison["comparison_label"] = np.select(
        [
            comparison["difference_from_sector"] > 0,
            comparison["difference_from_sector"] < 0,
        ],
        [
            "above sector average",
            "below sector average",
        ],
        default="equal to sector average",
    )

    trend = _classify_weekly_trend(
        uploaded_business_weekly["weekly_growth"],
        threshold=growth_threshold,
    )
    uploaded_business_weekly["trend"] = trend
    comparison["trend"] = trend

    latest_comparison = comparison.dropna(subset=["business_engagement_rate"]).tail(1)
    if latest_comparison.empty:
        latest_difference = 0
        latest_comparison_label = "equal to sector average"
        latest_week = pd.NaT
        latest_business_rate = np.nan
        latest_sector_rate = np.nan
        latest_weekly_growth = 0
    else:
        latest_row = latest_comparison.iloc[0]
        latest_difference = latest_row["difference_from_sector"]
        if pd.isna(latest_difference):
            latest_difference = 0
        latest_comparison_label = latest_row["comparison_label"]
        latest_week = latest_row["week"]
        latest_business_rate = latest_row["business_engagement_rate"]
        latest_sector_rate = latest_row["sector_engagement_rate"]
        latest_weekly_growth = latest_row["weekly_growth"]

    recommendation_message = _build_recommendation_message(
        business_name,
        sector,
        trend,
        latest_comparison_label,
        latest_difference,
    )

    summary = pd.DataFrame(
        [
            {
                "business_name": business_name,
                "sector": sector,
                "latest_week": latest_week,
                "latest_business_engagement_rate": latest_business_rate,
                "latest_sector_engagement_rate": latest_sector_rate,
                "difference_from_sector": latest_difference,
                "comparison_label": latest_comparison_label,
                "weekly_growth": latest_weekly_growth,
                "trend": trend,
                "recommendation_message": recommendation_message,
            }
        ]
    )

    safe_business_name = (
        str(business_name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    chart_path = output_dir / f"{safe_business_name}_vs_sector_ui_trend.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        sector_weekly["week"],
        sector_weekly["sector_engagement_rate"],
        color="tab:blue",
        marker="o",
        linewidth=2,
        label=f"{sector} sector weekly engagement rate",
    )
    ax.plot(
        uploaded_business_weekly["week"],
        uploaded_business_weekly["business_engagement_rate"],
        color="tab:orange",
        marker="o",
        linewidth=2,
        label=f"{business_name} weekly engagement rate",
    )
    ax.set_title(f"{business_name} vs {sector} Weekly Engagement Rate", fontsize=15)
    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Engagement Rate", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate(rotation=35)
    plt.tight_layout()
    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    sector_weekly.to_csv(output_dir / "sector_weekly_trends.csv", index=False)
    uploaded_business_weekly.to_csv(
        output_dir / "uploaded_business_weekly_trends.csv",
        index=False,
    )
    comparison.to_csv(output_dir / "business_vs_sector_momentum.csv", index=False)
    summary.to_csv(output_dir / "ui_business_momentum_summary.csv", index=False)

    return {
        "sector_weekly_trends": sector_weekly,
        "uploaded_business_weekly_trends": uploaded_business_weekly,
        "business_vs_sector_momentum": comparison,
        "ui_business_momentum_summary": summary,
        "chart_path": chart_path,
    }

# Sector Weekly Trends
# This answers:
# Is each sector improving, declining, or stable over time?

sector_weekly = (
    df.groupby(["sector", "week"], as_index=False)
    .agg(
        total_engagement=("engagement", "sum"),
        avg_engagement_rate=("engagement_rate", "mean"),
        total_likes=("likes_count", "sum"),
        total_comments=("comments_count", "sum"),
        total_views=("views_count", "sum"),
        total_posts=("business_name", "size"),
        active_businesses=("business_name", "nunique"),
    )
    .sort_values(["sector", "week"])
)

sector_weekly["sector_growth"] = (
    sector_weekly.groupby("sector")["avg_engagement_rate"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

sector_weekly["sector_trend_class"] = sector_weekly["sector_growth"].apply(
    lambda x: classify_trend(x, threshold=0.10)
)

# Latest sector status
sector_momentum = (
    sector_weekly.groupby("sector", as_index=False)
    .tail(1)[
        [
            "sector",
            "week",
            "avg_engagement_rate",
            "sector_growth",
            "sector_trend_class",
            "total_posts",
            "active_businesses",
        ]
    ]
    .rename(
        columns={
            "avg_engagement_rate": "latest_sector_engagement_rate",
            "sector_growth": "latest_sector_growth",
            "sector_trend_class": "sector_momentum_class",
        }
    )
)


# Business Weekly Trends

# This answers:
# Is each business improving, declining, stable, or inconsistent?

business_weekly = (
    df.groupby(["sector", "business_name", "week"], as_index=False)
    .agg(
        engagement_rate=("engagement_rate", "mean"),
        engagement=("engagement", "mean"),
        posts_count=("business_name", "size"),
        likes_count=("likes_count", "sum"),
        comments_count=("comments_count", "sum"),
        views_count=("views_count", "sum"),
    )
    .sort_values(["sector", "business_name", "week"])
)

ROLLING_WINDOW = 3
GROWTH_THRESHOLD = 0.10

business_weekly["rolling_engagement_rate"] = (
    business_weekly.groupby("business_name")["engagement_rate"]
    .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean())
)

business_weekly["business_growth"] = (
    business_weekly.groupby("business_name")["rolling_engagement_rate"]
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

business_weekly["trend_class"] = business_weekly["business_growth"].apply(
    lambda x: classify_trend(x, threshold=GROWTH_THRESHOLD)
)

# Detect inconsistent businesses
state_counts = (
    business_weekly.groupby("business_name")["trend_class"]
    .nunique()
    .rename("n_states")
)

business_weekly = business_weekly.merge(
    state_counts,
    on="business_name",
    how="left"
)

business_weekly["final_trend_class"] = np.where(
    business_weekly["n_states"] >= 3,
    "inconsistent",
    business_weekly["trend_class"]
)

business_momentum = (
    business_weekly.groupby(["sector", "business_name"], as_index=False)
    .tail(1)[
        [
            "sector",
            "business_name",
            "week",
            "rolling_engagement_rate",
            "business_growth",
            "final_trend_class",
            "posts_count",
        ]
    ]
    .rename(
        columns={
            "rolling_engagement_rate": "latest_rolling_engagement_rate",
            "business_growth": "latest_business_growth",
            "final_trend_class": "business_momentum_class",
        }
    )
)

# 6) Compare Business With Sector
# This gives real business value: Is the business performing better or worse than its sector?

comparison = business_momentum.merge(
    sector_momentum[
        [
            "sector",
            "latest_sector_engagement_rate",
            "sector_momentum_class",
        ]
    ],
    on="sector",
    how="left"
)

comparison["performance_vs_sector"] = np.where(
    comparison["latest_rolling_engagement_rate"]
    > comparison["latest_sector_engagement_rate"],
    "above_sector_average",
    "below_sector_average",
)

if __name__ == "__main__":
    # Simple Visualizations

    # Plot 1: Sector engagement rate over time Separate weekly engagement trend for each sector
    sectors = sector_weekly["sector"].unique()
    num_sectors = len(sectors)

    cols = 1
    rows = num_sectors

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))

    if num_sectors == 1:
        axes = [axes]

    for i, sector in enumerate(sectors):
        temp = sector_weekly[sector_weekly["sector"] == sector].copy()

        axes[i].plot(
            temp["week"],
            temp["avg_engagement_rate"],
            marker="o"
        )

        axes[i].set_title(f"{sector} Weekly Engagement Trend", fontsize=14)
        axes[i].set_xlabel("Week", fontsize=11)
        axes[i].set_ylabel("Avg Engagement Rate", fontsize=11)

        step = max(1, len(temp) // 6)
        axes[i].set_xticks(range(0, len(temp), step))
        axes[i].set_xticklabels(
            temp["week"].iloc[::step],
            rotation=35,
            ha="right",
            fontsize=9
        )

    plt.tight_layout(h_pad=3)

    fig.savefig(
        OUTPUTS_DIR / "sector_weekly_engagement_trends_separate.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    # Plot 2: Number of businesses by momentum class
    momentum_counts = (
        business_momentum["business_momentum_class"]
        .value_counts()
        .reset_index()
    )

    momentum_counts.columns = ["momentum_class", "business_count"]

    plt.figure(figsize=(8, 5))
    plt.bar(
        momentum_counts["momentum_class"],
        momentum_counts["business_count"],
    )

    plt.title("Number of Businesses by Momentum Class")
    plt.xlabel("Momentum Class")
    plt.ylabel("Number of Businesses")
    plt.tight_layout()

    plt.savefig(
        OUTPUTS_DIR / "business_momentum_class_counts.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Save Outputs
    sector_weekly.to_csv(
        OUTPUTS_DIR / "sector_weekly_trends.csv",
        index=False,
    )

    business_weekly.to_csv(
        OUTPUTS_DIR / "business_weekly_trends.csv",
        index=False,
    )

    business_momentum.to_csv(
        OUTPUTS_DIR / "business_momentum.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUTS_DIR / "business_vs_sector_momentum.csv",
        index=False,
    )

    # Final Insight
    print("Business Momentum Weekly Trends completed successfully.")
    print()
    print("Generated outputs:")
    print("- sector_weekly_trends.csv")
    print("- sector_momentum.csv")
    print("- business_weekly_trends.csv")
    print("- business_momentum.csv")
    print("- business_vs_sector_momentum.csv")
    print()
    print("Main value:")
    print("1. Sector trends explain how each sector performs over time.")
    print("2. Business momentum explains whether each business is improving, declining, stable, or inconsistent.")
    print("3. Business vs sector comparison shows whether a business is above or below its sector average.")
