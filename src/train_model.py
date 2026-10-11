"""
src/train_model.py
==================
Machine Learning Training Pipeline for Digital Distraction Pattern Predictor.

Key Responsibilities:
1. Loads screen time dataset and engineers temporal & distraction features.
2. Performs stratified train/test split to prevent data leakage.
3. Implements Scikit-learn Pipeline with ColumnTransformer (StandardScaler + OneHotEncoder).
4. Trains and compares a Baseline Logistic Regression and a Random Forest Classifier.
5. Trains an Unsupervised K-Means model to discover natural behavioral archetypes.
6. Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Exports fitted models with Joblib and saves evaluation metrics to JSON.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def locate_data():
    """Finds dataset from common project relative locations."""
    candidates = [
        os.path.join("data", "screen_time_app_usage_dataset.csv"),
        "screen_time_app_usage_dataset.csv",
        os.path.join("..", "data", "screen_time_app_usage_dataset.csv"),
        os.path.join("..", "screen_time_app_usage_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Dataset 'screen_time_app_usage_dataset.csv' not found.")


def prepare_data(filepath):
    """Loads raw dataset, cleans records, and constructs target variable."""
    df = pd.read_csv(filepath)

    # Feature extraction from timestamps
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_late_night"] = ((df["hour"] >= 22) | (df["hour"] < 5)).astype(int)

    # Category Weights for Distraction Potential
    cat_weights = {
        "Social": 1.0,
        "Entertainment": 0.9,
        "Utilities": 0.3,
        "Productivity": 0.0,
    }
    cat_weight = df["category"].map(cat_weights).fillna(0.5)

    # Normalized Screen Time Score (capped at 90 minutes)
    time_score = np.clip(df["screen_time_min"] / 60.0, 0, 1.5)

    # Compulsive Usage Score (launches and taps)
    compulsion_score = (df["launches"] / 4.0) * 0.5 + (df["interactions"] / 8.0) * 0.5

    # Circadian disruption penalty
    night_penalty = df["is_late_night"] * 0.3

    # Composite Distraction Score
    distraction_score = (cat_weight * 1.5) + (time_score * 1.2) + (compulsion_score * 0.8) + night_penalty

    # Target: 3 transparent ordinal classes
    def categorize(score):
        if score <= 1.49:
            return "Low Distraction"
        elif score <= 2.27:
            return "Moderate Distraction"
        else:
            return "High Distraction"

    df["distraction_level"] = distraction_score.apply(categorize)
    return df


def train_and_evaluate(df, output_models_dir="models", figures_dir="notebooks_or_reports/figures"):
    """Builds, trains, evaluates, and exports the ML pipelines."""
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" 🤖 TRAINING DIGITAL DISTRACTION PREDICTION MODELS")
    print("=" * 70)

    # Select features and target
    num_features = ["screen_time_min", "launches", "interactions", "hour"]
    cat_features = ["category", "day_of_week"]
    all_features = num_features + cat_features
    target = "distraction_level"

    X = df[all_features]
    y = df[target]

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\n[1] Train/Test Split completed:")
    print(f"    - Training Set: {len(X_train)} samples")
    print(f"    - Testing Set : {len(X_test)} samples")
    print("    - Stratified class distribution preserved across splits.")

    # Scikit-learn Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )

    # 1. Baseline Model: Logistic Regression
    print("\n[2] Training Baseline Model (Multinomial Logistic Regression)...")
    baseline_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline_pipeline.fit(X_train, y_train)

    train_acc_base = accuracy_score(y_train, baseline_pipeline.predict(X_train))
    test_acc_base = accuracy_score(y_test, baseline_pipeline.predict(X_test))
    print(f"    - Baseline Train Accuracy: {train_acc_base * 100:.2f}%")
    print(f"    - Baseline Test Accuracy : {test_acc_base * 100:.2f}%")

    # 2. Main Model: Random Forest Classifier
    print("\n[3] Training Random Forest Classifier...")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    train_acc_rf = accuracy_score(y_train, rf_pipeline.predict(X_train))
    y_pred_rf = rf_pipeline.predict(X_test)
    test_acc_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf, average="weighted")
    recall_rf = recall_score(y_test, y_pred_rf, average="weighted")
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print(f"    - Random Forest Train Accuracy: {train_acc_rf * 100:.2f}%")
    print(f"    - Random Forest Test Accuracy : {test_acc_rf * 100:.2f}%")
    print(f"    - Random Forest Test Precision: {precision_rf * 100:.2f}%")
    print(f"    - Random Forest Test Recall   : {recall_rf * 100:.2f}%")
    print(f"    - Random Forest Test F1-Score : {f1_rf * 100:.2f}%")

    # Classification Report
    labels = ["Low Distraction", "Moderate Distraction", "High Distraction"]
    clf_report = classification_report(y_test, y_pred_rf, target_names=labels, output_dict=True)
    print("\n[4] Detailed Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=labels))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_rf, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False
    )
    plt.title("Confusion Matrix - Random Forest Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Distraction Label", fontsize=10)
    plt.ylabel("Actual Distraction Label", fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"    Saved confusion matrix figure: {cm_path}")

    # Feature Importance Extraction
    rf_model = rf_pipeline.named_steps["classifier"]
    encoded_cat_names = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_features).tolist()
    feature_names = num_features + encoded_cat_names
    importances = rf_model.feature_importances_

    feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(9, 5))
    sns.barplot(x="Importance", y="Feature", hue="Feature", data=feat_df.head(10), palette="viridis", legend=False)
    plt.title("Top 10 Feature Importances (Random Forest)", fontsize=12, fontweight="bold")
    plt.xlabel("Gini Importance Score", fontsize=10)
    plt.ylabel("Engineered Feature", fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(figures_dir, "feature_importances.png")
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"    Saved feature importances figure: {fi_path}")

    # 3. Unsupervised Pattern Discovery: K-Means Clustering
    print("\n[5] Training Unsupervised K-Means Behavioral Clustering...")
    behavioral_features = ["screen_time_min", "launches", "interactions", "hour"]
    scaler_cluster = StandardScaler()
    X_scaled_behav = scaler_cluster.fit_transform(df[behavioral_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled_behav)
    df["usage_cluster"] = cluster_labels

    # Compute cluster profiles
    cluster_profiles = df.groupby("usage_cluster")[behavioral_features].mean().round(2).to_dict(orient="index")
    # Identify archetypes: quick brief check, balanced active usage, deep heavy binge
    cluster_archetype_names = {
        0: "Archetype A: Focused / Low-Engagement Utility",
        1: "Archetype B: Balanced / Moderate Engagement",
        2: "Archetype C: Prolonged / High-Intensity Binge",
    }
    print("    Cluster Profiles (Mean Values):")
    for cl_id, prof in cluster_profiles.items():
        name = cluster_archetype_names.get(cl_id, f"Cluster {cl_id}")
        print(f"    - {name}: {prof}")

    # Save models using Joblib
    model_file = os.path.join(output_models_dir, "model.joblib")
    joblib.dump(rf_pipeline, model_file)
    print(f"\n[6] Exported primary model pipeline: {model_file}")

    kmeans_file = os.path.join(output_models_dir, "kmeans_model.joblib")
    joblib.dump({"scaler": scaler_cluster, "kmeans": kmeans, "features": behavioral_features}, kmeans_file)
    print(f"    Exported K-Means cluster pipeline: {kmeans_file}")

    # Save metrics JSON for Streamlit UI and reports
    metrics_data = {
        "models": {
            "Baseline (Logistic Regression)": {
                "train_accuracy": float(train_acc_base),
                "test_accuracy": float(test_acc_base),
            },
            "Random Forest Classifier": {
                "train_accuracy": float(train_acc_rf),
                "test_accuracy": float(test_acc_rf),
                "test_precision": float(precision_rf),
                "test_recall": float(recall_rf),
                "test_f1": float(f1_rf),
                "classification_report": clf_report,
                "top_features": feat_df.head(8).to_dict(orient="records"),
            }
        },
        "target_classes": labels,
        "input_features": {
            "numerical": num_features,
            "categorical": cat_features,
        },
        "cluster_profiles": {str(k): v for k, v in cluster_profiles.items()},
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }

    metrics_file = os.path.join(output_models_dir, "model_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"    Exported evaluation metrics JSON: {metrics_file}")

    print("\n" + "=" * 70)
    print(" ✅ MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return rf_pipeline, metrics_data


def main():
    try:
        csv_path = locate_data()
        df = prepare_data(csv_path)
        train_and_evaluate(df)
    except Exception as e:
        print(f"\n❌ Error during model training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
