# 05 Anomaly Detection
# Sector-Based Explainable Anomaly Detection

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

# Project Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

sys.path.insert(0, str(NOTEBOOKS_DIR))

from utils.utils import ensure_project_dirs

ensure_project_dirs()

KPI_PATH = PROJECT_ROOT / "data" / "processed" / "kpi_dataset.csv"

OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2) Load Dataset
# =========================

df = pd.read_csv(KPI_PATH, parse_dates=["post_date"])

required_columns = [
    "business_name",
    "sector",
    "post_date",
    "post_type",
    "engagement_rate",
    "view_rate",
    "comment_rate",
    "like_rate",
    "view_engagement_rate",
    "discount_percent",
    "hashtags_count",
    "emoji_count",
    "caption_length",
    "views_count",
    "likes_count",
    "comments_count",
    "promo_post",
    "posting_hour",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["business_name"] = df["business_name"].astype(str).str.strip()
df["sector"] = df["sector"].astype(str).str.strip()

# Features

features = [
    "engagement_rate",
    "view_rate",
    "comment_rate",
    "like_rate",
    "view_engagement_rate",
    "discount_percent",
    "hashtags_count",
    "emoji_count",
    "caption_length",
]

#  Helper Functions


def run_zscore(X, threshold):
    zmax = np.abs(X).max(axis=1)
    return np.where(zmax >= threshold, -1, 1)


def run_iforest(X, contamination):
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    return model.fit_predict(X)


def run_lof(X, n_neighbors, contamination=0.05):
    n_neighbors = min(n_neighbors, max(2, len(X) - 1))

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    return model.fit_predict(X)


def evaluate_method(y_pred, method_name):
    anomaly_count = int((y_pred == -1).sum())
    anomaly_ratio = anomaly_count / len(y_pred)

    target_ratio = 0.05
    balance_score = 1 - abs(anomaly_ratio - target_ratio)

    interpretability = {
        "zscore": 0.95,
        "isolation_forest": 0.80,
        "lof": 0.75,
    }.get(method_name, 0.70)

    final_score = (0.6 * balance_score) + (0.4 * interpretability)

    return anomaly_count, anomaly_ratio, balance_score, interpretability, final_score


def classify_anomaly_type(sector_df, y_pred):
    out = sector_df.copy()
    out["anomaly_label"] = y_pred

    sector_median = out["engagement_rate"].median()

    out["anomaly_type"] = np.where(
        (out["anomaly_label"] == -1)
        & (out["engagement_rate"] >= sector_median),
        "positive_anomaly",
        np.where(
            out["anomaly_label"] == -1,
            "negative_anomaly",
            "normal"
        )
    )

    return out


def add_anomaly_strength(anomalies_df):
    result = anomalies_df.copy()

    result["anomaly_strength"] = np.where(
        result["anomaly_type"] == "positive_anomaly",
        result["engagement_rate"],
        -result["engagement_rate"]
    )

    return result


def summarize_patterns(df_part, sector_name, anomaly_type):
    if df_part.empty:
        return {
            "sector": sector_name,
            "anomaly_type": anomaly_type,
            "count": 0,
            "avg_engagement_rate": np.nan,
            "avg_views": np.nan,
            "avg_likes": np.nan,
            "avg_comments": np.nan,
            "avg_caption_length": np.nan,
            "avg_hashtags": np.nan,
            "avg_emojis": np.nan,
            "most_common_post_type": None,
            "promo_post_ratio": np.nan,
            "most_common_posting_hour": None,
            "insight": "No enough anomalies detected for this group."
        }

    most_common_post_type = (
        df_part["post_type"].mode().iloc[0]
        if not df_part["post_type"].mode().empty
        else None
    )

    most_common_hour = (
        df_part["posting_hour"].mode().iloc[0]
        if not df_part["posting_hour"].mode().empty
        else None
    )

    promo_ratio = df_part["promo_post"].fillna(False).astype(bool).mean()

    if anomaly_type == "positive_anomaly":
        insight = (
            f"In {sector_name}, viral posts are commonly {most_common_post_type} posts, "
            f"with average engagement rate {df_part['engagement_rate'].mean():.4f}. "
            f"They tend to have average views {df_part['views_count'].mean():.0f}, "
            f"average caption length {df_part['caption_length'].mean():.0f}, "
            f"and promo usage ratio {promo_ratio:.2f}."
        )
    else:
        insight = (
            f"In {sector_name}, weak posts are commonly {most_common_post_type} posts, "
            f"with average engagement rate {df_part['engagement_rate'].mean():.4f}. "
            f"They tend to have average views {df_part['views_count'].mean():.0f}, "
            f"average caption length {df_part['caption_length'].mean():.0f}, "
            f"and promo usage ratio {promo_ratio:.2f}."
        )

    return {
        "sector": sector_name,
        "anomaly_type": anomaly_type,
        "count": len(df_part),
        "avg_engagement_rate": df_part["engagement_rate"].mean(),
        "avg_views": df_part["views_count"].mean(),
        "avg_likes": df_part["likes_count"].mean(),
        "avg_comments": df_part["comments_count"].mean(),
        "avg_caption_length": df_part["caption_length"].mean(),
        "avg_hashtags": df_part["hashtags_count"].mean(),
        "avg_emojis": df_part["emoji_count"].mean(),
        "most_common_post_type": most_common_post_type,
        "promo_post_ratio": promo_ratio,
        "most_common_posting_hour": most_common_hour,
        "insight": insight,
    }


def ensure_anomaly_type(df):
    result = df.copy()

    if "anomaly_type" in result.columns:
        return result

    if "anomaly_label" in result.columns:
        engagement_reference = result["engagement_rate"].median()

        result["anomaly_type"] = np.where(
            (result["anomaly_label"] == -1)
            & (result["engagement_rate"] >= engagement_reference),
            "positive_anomaly",
            np.where(
                result["anomaly_label"] == -1,
                "negative_anomaly",
                "normal"
            )
        )
    else:
        result["anomaly_type"] = "normal"

    return result


def add_behavior_features_from_available_columns(df):
    result = df.copy()

    if "posting_frequency" not in result.columns and "business_name" in result.columns:
        result["posting_frequency"] = result.groupby("business_name")["business_name"].transform("size")

    direct_mappings = {
        "avg_caption_length": "caption_length",
        "avg_hashtags_count": "hashtags_count",
        "avg_emoji_count": "emoji_count",
        "avg_engagement_rate": "engagement_rate",
        "avg_view_rate": "view_rate",
        "avg_comment_rate": "comment_rate",
    }

    for behavior_col, source_col in direct_mappings.items():
        if behavior_col not in result.columns and source_col in result.columns:
            result[behavior_col] = pd.to_numeric(result[source_col], errors="coerce")

    boolean_mappings = {
        "percentage_reels": "is_reel",
        "percentage_promo_posts": "promo_post",
        "percentage_CTA_posts": "CTA_present",
        "percentage_location_posts": "mentions_location",
        "percentage_religious_theme": "religious_theme",
        "percentage_patriotic_theme": "patriotic_theme",
        "percentage_arabic_dialect_style": "arabic_dialect_style",
    }

    for behavior_col, source_col in boolean_mappings.items():
        if behavior_col not in result.columns and source_col in result.columns:
            result[behavior_col] = result[source_col].fillna(False).astype(bool).astype(int)

    if "percentage_reels" not in result.columns and "post_type" in result.columns:
        result["percentage_reels"] = (
            result["post_type"].astype(str).str.lower().eq("reel").astype(int)
        )

    if "percentage_images" not in result.columns and "post_type" in result.columns:
        result["percentage_images"] = (
            result["post_type"].astype(str).str.lower().eq("image").astype(int)
        )

    if "percentage_carousels" not in result.columns and "post_type" in result.columns:
        result["percentage_carousels"] = (
            result["post_type"].astype(str).str.lower().eq("carousel").astype(int)
        )

    return result


def generate_anomaly_behavior_recommendations(df, output_dir, top_n=5):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_anomaly_type(df.copy())
    df = add_behavior_features_from_available_columns(df)

    behavior_cols = [
        "posting_frequency",
        "avg_caption_length",
        "avg_hashtags_count",
        "avg_emoji_count",
        "percentage_reels",
        "percentage_images",
        "percentage_carousels",
        "percentage_promo_posts",
        "percentage_CTA_posts",
        "percentage_location_posts",
        "percentage_religious_theme",
        "percentage_patriotic_theme",
        "percentage_arabic_dialect_style",
        "avg_engagement_rate",
        "avg_view_rate",
        "avg_comment_rate",
    ]

    friendly_names = {
        "posting_frequency": "post more consistently",
        "avg_caption_length": "adjust caption length",
        "avg_hashtags_count": "use hashtags more strategically",
        "avg_emoji_count": "use emojis more effectively",
        "percentage_reels": "use more reels",
        "percentage_images": "use more image posts",
        "percentage_carousels": "use more carousel posts",
        "percentage_promo_posts": "balance promotional content",
        "percentage_CTA_posts": "add stronger call-to-actions",
        "percentage_location_posts": "mention locations more often",
        "percentage_religious_theme": "use religious seasonal themes carefully",
        "percentage_patriotic_theme": "use patriotic themes when relevant",
        "percentage_arabic_dialect_style": "use Arabic dialect style when appropriate",
        "avg_engagement_rate": "improve engagement rate",
        "avg_view_rate": "improve video/view performance",
        "avg_comment_rate": "encourage more comments",
    }

    available_behavior_cols = [
        col for col in behavior_cols
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    profile_summary_path = output_dir / "anomaly_behavior_profile_summary.csv"
    recommendations_path = output_dir / "anomaly_recommendations.csv"
    chart_path = output_dir / "anomaly_recommendation_drivers.png"

    profile_columns = [
        "behavior_feature",
        "positive_mean",
        "negative_mean",
        "normal_mean",
        "positive_diff",
        "negative_diff",
        "positive_relative_percent",
        "negative_relative_percent",
    ]
    recommendation_columns = [
        "recommendation_type",
        "behavior_feature",
        "friendly_action",
        "direction",
        "relative_difference_percent",
        "recommendation_message",
    ]

    if not available_behavior_cols:
        warning_message = "No available behavior columns found for anomaly recommendations."
        print(f"Warning: {warning_message}")

        profile_summary_df = pd.DataFrame(columns=profile_columns)
        recommendations_df = pd.DataFrame(columns=recommendation_columns)
        recommendations_df.attrs["warning_message"] = warning_message

        profile_summary_df.to_csv(profile_summary_path, index=False)
        recommendations_df.to_csv(recommendations_path, index=False)

        return {
            "profile_summary_df": profile_summary_df,
            "recommendations_df": recommendations_df,
            "profile_summary_path": profile_summary_path,
            "recommendations_path": recommendations_path,
            "chart_path": None,
        }

    positive_df = df[df["anomaly_type"] == "positive_anomaly"]
    negative_df = df[df["anomaly_type"] == "negative_anomaly"]
    normal_df = df[df["anomaly_type"] == "normal"]

    positive_means = positive_df[available_behavior_cols].mean()
    negative_means = negative_df[available_behavior_cols].mean()
    normal_means = normal_df[available_behavior_cols].mean()

    positive_diff = positive_means - normal_means
    negative_diff = negative_means - normal_means

    positive_relative = (
        (positive_means - normal_means) / (normal_means + 1e-6)
    ) * 100
    negative_relative = (
        (negative_means - normal_means) / (normal_means + 1e-6)
    ) * 100

    profile_summary_df = pd.DataFrame({
        "behavior_feature": available_behavior_cols,
        "positive_mean": positive_means.reindex(available_behavior_cols).values,
        "negative_mean": negative_means.reindex(available_behavior_cols).values,
        "normal_mean": normal_means.reindex(available_behavior_cols).values,
        "positive_diff": positive_diff.reindex(available_behavior_cols).values,
        "negative_diff": negative_diff.reindex(available_behavior_cols).values,
        "positive_relative_percent": positive_relative.reindex(available_behavior_cols).values,
        "negative_relative_percent": negative_relative.reindex(available_behavior_cols).values,
    })

    warning_messages = []
    if positive_df.empty:
        warning_messages.append("No positive anomalies found.")
    if negative_df.empty:
        warning_messages.append("No negative anomalies found.")
    if normal_df.empty:
        warning_messages.append("No normal businesses/posts found for baseline comparison.")

    if warning_messages:
        warning_message = " ".join(warning_messages)
        print(f"Warning: {warning_message}")

        recommendations_df = pd.DataFrame(columns=recommendation_columns)
        recommendations_df.attrs["warning_message"] = warning_message

        profile_summary_df.to_csv(profile_summary_path, index=False)
        recommendations_df.to_csv(recommendations_path, index=False)

        return {
            "profile_summary_df": profile_summary_df,
            "recommendations_df": recommendations_df,
            "profile_summary_path": profile_summary_path,
            "recommendations_path": recommendations_path,
            "chart_path": None,
        }

    recommendation_rows = []

    top_positive = (
        positive_relative.dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )

    for feature, relative_value in top_positive.items():
        if relative_value <= 0:
            continue

        friendly_action = friendly_names.get(feature, feature.replace("_", " "))
        recommendation_rows.append({
            "recommendation_type": "do_more",
            "behavior_feature": feature,
            "friendly_action": friendly_action,
            "direction": "higher_than_normal",
            "relative_difference_percent": relative_value,
            "recommendation_message": (
                f"Do more: {friendly_action}. "
                f"Positive anomalies are higher than normal by about {relative_value:.1f}%."
            ),
        })

    top_negative = (
        negative_relative.dropna()
        .sort_values(ascending=True)
        .head(top_n)
    )

    for feature, relative_value in top_negative.items():
        if relative_value >= 0:
            continue

        friendly_action = friendly_names.get(feature, feature.replace("_", " "))
        recommendation_rows.append({
            "recommendation_type": "avoid_or_reduce",
            "behavior_feature": feature,
            "friendly_action": friendly_action,
            "direction": "lower_than_normal",
            "relative_difference_percent": relative_value,
            "recommendation_message": (
                f"Avoid/reduce: weak behavior in {feature}. "
                f"Negative anomalies are lower than normal by about {abs(relative_value):.1f}%."
            ),
        })

    recommendations_df = pd.DataFrame(
        recommendation_rows,
        columns=recommendation_columns,
    )

    profile_summary_df.to_csv(profile_summary_path, index=False)
    recommendations_df.to_csv(recommendations_path, index=False)

    if recommendations_df.empty:
        chart_path = None
    else:
        chart_df = recommendations_df.copy()
        chart_df["plot_value"] = chart_df["relative_difference_percent"]
        chart_df["label"] = chart_df["friendly_action"]
        chart_df = chart_df.sort_values("plot_value")

        colors = np.where(
            chart_df["recommendation_type"] == "do_more",
            "tab:green",
            "tab:red"
        )

        fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(chart_df))))
        ax.barh(chart_df["label"], chart_df["plot_value"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Top Anomaly Recommendation Drivers", fontsize=14)
        ax.set_xlabel("Relative Difference vs Normal (%)", fontsize=11)
        ax.set_ylabel("Behavior Driver", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        fig.savefig(chart_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "profile_summary_df": profile_summary_df,
        "recommendations_df": recommendations_df,
        "profile_summary_path": profile_summary_path,
        "recommendations_path": recommendations_path,
        "chart_path": chart_path,
    }


def run_sector_based_anomaly_detection():
    all_anomalies = []
    all_sector_results = []
    top_positive_rows = []
    top_negative_rows = []
    experiment_rows = []
    pattern_rows = []

    sectors = df["sector"].dropna().unique()

    for sector in sectors:
        sector_df = df[df["sector"] == sector].copy()

        if len(sector_df) < 10:
            print(f"Skipping sector {sector}: not enough data.")
            continue

        X_raw = sector_df[features].fillna(0)
        X = StandardScaler().fit_transform(X_raw)

        method_outputs = {}

        for threshold in [2, 2.5, 3]:
            y = run_zscore(X, threshold)
            method_key = f"zscore_{threshold}"

            method_outputs[method_key] = {
                "method": "zscore",
                "setting": f"threshold={threshold}",
                "y": y,
            }

        for contamination in [0.03, 0.05, 0.10]:
            y = run_iforest(X, contamination)
            method_key = f"iforest_{contamination}"

            method_outputs[method_key] = {
                "method": "isolation_forest",
                "setting": f"contamination={contamination}",
                "y": y,
            }

        for n_neighbors in [10, 20, 35]:
            y = run_lof(X, n_neighbors, contamination=0.05)
            method_key = f"lof_{n_neighbors}"

            method_outputs[method_key] = {
                "method": "lof",
                "setting": f"n_neighbors={n_neighbors}",
                "y": y,
            }

        for method_key, info in method_outputs.items():
            anomaly_count, anomaly_ratio, balance_score, interpretability, final_score = evaluate_method(
                info["y"],
                info["method"]
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

        fixed_method_key = "iforest_0.05"
        fixed_method_info = method_outputs[fixed_method_key]
        best_y = fixed_method_info["y"]

        sector_result = classify_anomaly_type(sector_df, best_y)
        sector_result["best_method"] = fixed_method_info["method"]
        sector_result["best_setting"] = fixed_method_info["setting"]
        sector_result["anomaly_detection_scope"] = "sector_based"

        all_sector_results.append(sector_result)

        sector_anomalies = sector_result[
            sector_result["anomaly_label"] == -1
        ].copy()

        sector_anomalies = add_anomaly_strength(sector_anomalies)

        all_anomalies.append(sector_anomalies)

        positive = (
            sector_anomalies[
                sector_anomalies["anomaly_type"] == "positive_anomaly"
            ]
            .sort_values("engagement_rate", ascending=False)
            .head(5)
        )

        negative = (
            sector_anomalies[
                sector_anomalies["anomaly_type"] == "negative_anomaly"
            ]
            .sort_values("engagement_rate", ascending=True)
            .head(5)
        )

        top_positive_rows.append(positive)
        top_negative_rows.append(negative)

        pattern_rows.append(
            summarize_patterns(positive, sector, "positive_anomaly")
        )

        pattern_rows.append(
            summarize_patterns(negative, sector, "negative_anomaly")
        )


    # =========================
    # 6) Combine Outputs
    # =========================
    anomaly_experiments = pd.DataFrame(experiment_rows)

    all_anomalies_df = (
        pd.concat(all_anomalies, ignore_index=True)
        if all_anomalies
        else pd.DataFrame()
    )

    all_anomaly_results_df = (
        pd.concat(all_sector_results, ignore_index=True)
        if all_sector_results
        else pd.DataFrame()
    )

    top_positive_anomalies = (
        pd.concat(top_positive_rows, ignore_index=True)
        if top_positive_rows
        else pd.DataFrame()
    )

    top_negative_anomalies = (
        pd.concat(top_negative_rows, ignore_index=True)
        if top_negative_rows
        else pd.DataFrame()
    )

    sector_pattern_summary = pd.DataFrame(pattern_rows)

    top_anomalies_for_client = pd.concat(
        [top_positive_anomalies, top_negative_anomalies],
        ignore_index=True
    )

    client_columns = [
        "sector",
        "business_name",
        "post_date",
        "post_type",
        "engagement_rate",
        "views_count",
        "likes_count",
        "comments_count",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "promo_post",
        "posting_hour",
        "anomaly_type",
        "best_method",
        "best_setting",
        "anomaly_detection_scope",
    ]

    top_anomalies_for_client = top_anomalies_for_client[
        [col for col in client_columns if col in top_anomalies_for_client.columns]
    ]

    # Save Outputs
    all_anomalies_df.to_csv(
        OUTPUTS_DIR / "sector_based_anomalies.csv",
        index=False
    )

    top_positive_anomalies.to_csv(
        OUTPUTS_DIR / "top5_positive_anomalies_by_sector.csv",
        index=False
    )

    top_negative_anomalies.to_csv(
        OUTPUTS_DIR / "top5_negative_anomalies_by_sector.csv",
        index=False
    )

    top_anomalies_for_client.to_csv(
        OUTPUTS_DIR / "top_anomalies_for_client.csv",
        index=False
    )

    sector_pattern_summary.to_csv(
        OUTPUTS_DIR / "sector_anomaly_pattern_summary.csv",
        index=False
    )

    anomaly_experiments.to_csv(
        OUTPUTS_DIR / "sector_anomaly_experiments.csv",
        index=False
    )

    anomaly_recommendation_outputs = generate_anomaly_behavior_recommendations(
        all_anomaly_results_df,
        OUTPUTS_DIR,
    )

    # Visualizations
    comparison_df = anomaly_experiments.copy()

    pivot_scores = comparison_df.pivot_table(
        index="sector",
        columns="method",
        values="final_score",
        aggfunc="mean"
    )

    pivot_scores.plot(
        kind="bar",
        figsize=(14, 6),
        width=0.8
    )

    plt.title("Anomaly Detection Algorithm Comparison by Sector")
    plt.xlabel("Sector")
    plt.ylabel("Final Score")
    plt.xticks(rotation=35, ha="right")
    plt.legend(title="Algorithm")
    plt.tight_layout()

    plt.savefig(
        OUTPUTS_DIR / "algorithm_comparison_by_sector.png"
    )

    plt.show()

    if not all_anomalies_df.empty:
        anomaly_counts = (
            all_anomalies_df.groupby(["sector", "anomaly_type"])
            .size()
            .reset_index(name="count")
        )

        pivot_counts = anomaly_counts.pivot(
            index="sector",
            columns="anomaly_type",
            values="count"
        ).fillna(0)

        pivot_counts.plot(kind="bar", figsize=(12, 6))
        plt.title("Positive vs Negative Anomalies by Sector")
        plt.xlabel("Sector")
        plt.ylabel("Number of Anomalies")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "positive_negative_anomalies_by_sector.png")
        plt.show()

    # Final Summary
    print("Sector-Based Explainable Anomaly Detection completed successfully.")
    print()
    print("Saved outputs to:")
    print(OUTPUTS_DIR)
    print()
    print("Generated files:")
    print("- sector_based_anomalies.csv")
    print("- top5_positive_anomalies_by_sector.csv")
    print("- top5_negative_anomalies_by_sector.csv")
    print("- top_anomalies_for_client.csv")
    print("- sector_anomaly_pattern_summary.csv")
    print("- sector_anomaly_experiments.csv")
    print("- anomaly_behavior_profile_summary.csv")
    print("- anomaly_recommendations.csv")
    print("- anomaly_recommendation_drivers.png (when recommendations exist)")
    print("- algorithm_comparison_by_sector.png")
    print("- positive_negative_anomalies_by_sector.png")


if __name__ == "__main__":
    run_sector_based_anomaly_detection()
