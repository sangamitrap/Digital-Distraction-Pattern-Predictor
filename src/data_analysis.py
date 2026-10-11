"""
src/data_analysis.py
====================
Exploratory Data Analysis (EDA) module for Digital Distraction Pattern Predictor.

This script:
1. Loads the screen time and app usage dataset.
2. Checks dimensions, column types, missing values, duplicates, and distributions.
3. Detects noise columns and missing patterns.
4. Formulates a transparent, domain-justified Distraction Pattern Index.
5. Generates and saves informative visual plots to 'notebooks_or_reports/figures/'.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_dataset_path():
    """Locate the dataset file gracefully in common relative project paths."""
    candidates = [
        os.path.join("data", "screen_time_app_usage_dataset.csv"),
        "screen_time_app_usage_dataset.csv",
        os.path.join("..", "data", "screen_time_app_usage_dataset.csv"),
        os.path.join("..", "screen_time_app_usage_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find 'screen_time_app_usage_dataset.csv'. "
        "Please ensure the file is in 'data/' or the project root directory."
    )


def load_and_inspect_data(filepath):
    """Loads CSV dataset and prints comprehensive summary statistics."""
    print("=" * 70)
    print(" 📊 DIGITAL DISTRACTION PATTERN PREDICTOR - DATASET INSPECTION")
    print("=" * 70)
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    print(f"\n[1] Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[2] Column Names & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"    - {col:<20} : {dtype}")

    # Duplicate check
    duplicate_count = df.duplicated().sum()
    print(f"\n[3] Duplicate Rows: {duplicate_count}")

    # Missing values check
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print("\n[4] Missing Values Check:")
    if missing_cols.empty:
        print("    No missing values found across all columns.")
    else:
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"    - {col:<20} : {count} missing ({pct:.1f}%)")
        print("    ℹ️ Note: 'youtube_*' metrics are only populated for YouTube app rows.")

    # Target Candidates Evaluation
    print("\n[5] Target Candidate & Domain Analysis:")
    print("    - 'category' distribution:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"        * {cat:<15}: {cnt} sessions ({cnt/len(df)*100:.1f}%)")

    print("\n    - 'is_productive' distribution:")
    for prod, cnt in df["is_productive"].value_counts().items():
        print(f"        * {str(prod):<15}: {cnt} sessions ({cnt/len(df)*100:.1f}%)")

    # Extra synthetic columns notice
    extra_cols = [c for c in df.columns if c.startswith("extra_col_")]
    if extra_cols:
        print(f"\n    - Detected {len(extra_cols)} benchmark noise columns: {extra_cols[0]} to {extra_cols[-1]}")
        print("      These contain Gaussian N(50, 10^2) noise without domain relevance and will be excluded.")

    return df


def engineer_features(df):
    """
    Cleans data, extracts temporal features, and computes the transparent
    domain-justified Distraction Level.
    """
    df_clean = df.copy()

    # Convert date to datetime
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean["hour"] = df_clean["date"].dt.hour
    df_clean["day_of_week"] = df_clean["date"].dt.day_name()
    df_clean["is_weekend"] = df_clean["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df_clean["is_late_night"] = ((df_clean["hour"] >= 22) | (df_clean["hour"] < 5)).astype(int)

    # Domain Weight for Category
    # Social and Entertainment apps are algorithmically designed for engagement loops (higher distraction weight).
    # Utilities (Maps, Camera, Weather) are utility tasks.
    # Productivity (Slack, Docs, Teams, Notion) are work/study tasks.
    category_weights = {
        "Social": 1.0,
        "Entertainment": 0.9,
        "Utilities": 0.3,
        "Productivity": 0.0,
    }
    cat_weight = df_clean["category"].map(category_weights).fillna(0.5)

    # Normalized Screen Time Score (capped at 90 minutes)
    time_score = np.clip(df_clean["screen_time_min"] / 60.0, 0, 1.5)

    # Launch & Interaction Compulsion Score
    compulsion_score = (df_clean["launches"] / 4.0) * 0.5 + (df_clean["interactions"] / 8.0) * 0.5

    # Late Night Context Modifier (+0.3 penalty for circadian rhythm disruption)
    night_penalty = df_clean["is_late_night"] * 0.3

    # Composite Distraction Index
    distraction_score = (cat_weight * 1.5) + (time_score * 1.2) + (compulsion_score * 0.8) + night_penalty
    df_clean["distraction_score"] = distraction_score

    # Transparent classification into 3 balanced categories:
    # Low Distraction: <= 1.49
    # Moderate Distraction: 1.49 to 2.27
    # High Distraction: > 2.27
    def categorize(score):
        if score <= 1.49:
            return "Low Distraction"
        elif score <= 2.27:
            return "Moderate Distraction"
        else:
            return "High Distraction"

    df_clean["distraction_level"] = distraction_score.apply(categorize)
    return df_clean


def generate_visualizations(df, output_dir="notebooks_or_reports/figures"):
    """Creates and saves analytical visualizations for exploratory report."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"Low Distraction": "#2ecc71", "Moderate Distraction": "#f39c12", "High Distraction": "#e74c3c"}

    print("\n[6] Generating Analytical Visualizations...")

    # 1. Screen Time Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["screen_time_min"], bins=40, kde=True, color="#3498db", edgecolor="none")
    plt.axvline(df["screen_time_min"].median(), color="#e74c3c", linestyle="--", label=f"Median ({df['screen_time_min'].median():.1f} min)")
    plt.title("Distribution of Screen Time per Session (Minutes)", fontsize=13, fontweight="bold")
    plt.xlabel("Screen Time (min)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "distribution_screen_time.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"    Saved: {plot1_path}")

    # 2. Screen Time by App Category
    plt.figure(figsize=(9, 5))
    cat_palette = {"Social": "#9b59b6", "Entertainment": "#e67e22", "Productivity": "#27ae60", "Utilities": "#3498db"}
    sns.boxplot(x="category", y="screen_time_min", hue="category", data=df, palette=cat_palette, showmeans=True, boxprops=dict(alpha=0.85), legend=False)
    plt.title("Screen Time Variation Across App Categories", fontsize=13, fontweight="bold")
    plt.xlabel("Application Category", fontsize=11)
    plt.ylabel("Screen Time (minutes)", fontsize=11)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "category_screen_time.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"    Saved: {plot2_path}")

    # 3. Distraction Level Distribution
    plt.figure(figsize=(8, 5))
    order = ["Low Distraction", "Moderate Distraction", "High Distraction"]
    ax = sns.countplot(x="distraction_level", hue="distraction_level", data=df, order=order, palette=palette, edgecolor="black", linewidth=0.5, legend=False)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    plt.title("Target Distribution: Digital Distraction Level", fontsize=13, fontweight="bold")
    plt.xlabel("Distraction Pattern Category", fontsize=11)
    plt.ylabel("Session Count", fontsize=11)
    plt.tight_layout()
    plot3_path = os.path.join(output_dir, "distraction_level_distribution.png")
    plt.savefig(plot3_path, dpi=200)
    plt.close()
    print(f"    Saved: {plot3_path}")

    # 4. Launches vs Interactions by Category
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        x="launches",
        y="interactions",
        hue="category",
        data=df,
        palette=cat_palette,
        alpha=0.6,
        s=45
    )
    plt.title("App Launches vs. In-App User Interactions", fontsize=13, fontweight="bold")
    plt.xlabel("App Launches (Count)", fontsize=11)
    plt.ylabel("Interactions (Taps/Clicks)", fontsize=11)
    plt.tight_layout()
    plot4_path = os.path.join(output_dir, "launches_vs_interactions.png")
    plt.savefig(plot4_path, dpi=200)
    plt.close()
    print(f"    Saved: {plot4_path}")

    # 5. Usage Patterns by Hour of Day
    hourly = df.groupby("hour")[["screen_time_min", "launches"]].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(hourly["hour"], hourly["screen_time_min"], marker="o", color="#2980b9", linewidth=2.5, label="Mean Screen Time (min)")
    plt.axvspan(22, 24, alpha=0.15, color="red", label="Late Night Window (22:00 - 05:00)")
    plt.axvspan(0, 5, alpha=0.15, color="red")
    plt.title("Diurnal Pattern: Screen Time Duration Across 24 Hours", fontsize=13, fontweight="bold")
    plt.xlabel("Hour of Day (0 to 23)", fontsize=11)
    plt.ylabel("Average Screen Time (min)", fontsize=11)
    plt.xticks(range(0, 24, 2))
    plt.legend()
    plt.tight_layout()
    plot5_path = os.path.join(output_dir, "usage_by_hour.png")
    plt.savefig(plot5_path, dpi=200)
    plt.close()
    print(f"    Saved: {plot5_path}")

    print("\n✅ All exploratory charts successfully generated and saved!")


def main():
    try:
        csv_path = find_dataset_path()
        df = load_and_inspect_data(csv_path)
        df_featured = engineer_features(df)
        generate_visualizations(df_featured)
        print("\n" + "=" * 70)
        print(" EDA Complete! Review 'notebooks_or_reports/figures/' for generated charts.")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error during exploratory data analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
