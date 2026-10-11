"""
app.py
======
Digital Distraction Pattern Predictor - Streamlit Web Application

Features:
- Live Interactive Predictor with real-time confidence scores and wellbeing tips.
- Exploratory Analytics Dashboard with interactive charts.
- Model Performance & Viva Showcase comparing Baseline and Random Forest.
- Unsupervised Behavioral Archetypes (K-Means clustering breakdown).
- Ethical AI & Academic Transparency disclaimers.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Digital Distraction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Glassmorphism, Modern Fonts, Vibrant Theme)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Gradient header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2050 50%, #1a365d 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 2.2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #60a5fa, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.5;
        max-width: 900px;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.25rem;
    }

    /* Prediction Result Cards */
    .result-card {
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
        color: #ffffff;
    }
    .result-high {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2) 0%, rgba(153, 27, 27, 0.4) 100%);
        border: 1.5px solid #ef4444;
    }
    .result-mod {
        background: linear-gradient(135deg, rgba(217, 119, 6, 0.2) 0%, rgba(180, 83, 9, 0.4) 100%);
        border: 1.5px solid #f59e0b;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.4) 100%);
        border: 1.5px solid #10b981;
    }

    .badge-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Tab enhancements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Resource Loading Helpers (with Streamlit caching)
# ---------------------------------------------------------
@st.cache_resource
def load_trained_pipeline():
    """Load the trained machine learning pipeline and metrics."""
    model_paths = ["models/model.joblib", "../models/model.joblib"]
    pipeline = None
    for p in model_paths:
        if os.path.exists(p):
            pipeline = joblib.load(p)
            break

    metrics_paths = ["models/model_metrics.json", "../models/model_metrics.json"]
    metrics = None
    for p in metrics_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            break

    return pipeline, metrics


@st.cache_data
def load_dataset():
    """Load the dataset safely."""
    data_paths = [
        "data/screen_time_app_usage_dataset.csv",
        "screen_time_app_usage_dataset.csv",
        "../data/screen_time_app_usage_dataset.csv"
    ]
    for p in data_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["date"] = pd.to_datetime(df["date"])
            df["hour"] = df["date"].dt.hour
            df["day_of_week"] = df["date"].dt.day_name()
            df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
            return df
    return None


# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">📱 Digital Distraction Pattern Predictor</div>
    <div class="hero-subtitle">
        An explainable Machine Learning application that analyzes digital device usage habits,
        screen time duration, app categories, and diurnal patterns to identify and predict digital distraction behaviors.
    </div>
</div>
""", unsafe_allow_html=True)

pipeline, metrics = load_trained_pipeline()
df = load_dataset()

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_analytics, tab_model, tab_clusters, tab_about = st.tabs([
    "🎯 Live Predictor",
    "📊 Dataset & Analytics",
    "🤖 Model Performance & Viva",
    "🔍 Behavioral Archetypes",
    "ℹ️ Project & Ethics"
])

# =========================================================
# TAB 1: LIVE PREDICTOR
# =========================================================
with tab_predict:
    st.subheader("Predict Distraction Pattern for a Digital Session")
    st.markdown(
        "Enter the digital usage session details below. The trained **Random Forest Pipeline** "
        "will preprocess features, evaluate interactions, and predict the distraction pattern level with full confidence breakdown."
    )

    col_in1, col_in2 = st.columns([1, 1], gap="large")

    with col_in1:
        st.markdown("#### 1️⃣ Application Context")
        category_choice = st.selectbox(
            "Application Category",
            options=["Social", "Entertainment", "Utilities", "Productivity"],
            help="Category of the app (e.g., Social = Instagram/TikTok, Productivity = Slack/Docs)",
            key="input_category"
        )

        app_examples = {
            "Social": "Instagram, TikTok, WhatsApp, Snapchat, Twitter, Facebook",
            "Entertainment": "YouTube, Netflix, Spotify, Twitch",
            "Productivity": "Slack, Google Docs, Notion, Microsoft Teams, Evernote",
            "Utilities": "Chrome, Camera, Google Maps, Gmail, Weather"
        }
        st.caption(f"💡 Common examples: *{app_examples[category_choice]}*")

        day_choice = st.selectbox(
            "Day of the Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4,
            key="input_day"
        )

        hour_choice = st.slider(
            "Hour of Day (24-Hour Format)",
            min_value=0,
            max_value=23,
            value=22,
            format="%d:00",
            help="Session start time. Hours between 22:00 and 05:00 represent night usage.",
            key="input_hour"
        )
        if hour_choice >= 22 or hour_choice < 5:
            st.warning("🌙 Late-night session detected (higher risk of sleep disruption).")

    with col_in2:
        st.markdown("#### 2️⃣ Behavioral Usage Metrics")
        screen_time_choice = st.slider(
            "Session Screen Time (Minutes)",
            min_value=1.0,
            max_value=180.0,
            value=45.0,
            step=1.0,
            help="Total consecutive minutes spent actively on this app.",
            key="input_time"
        )

        launches_choice = st.number_input(
            "App Launches Count",
            min_value=0,
            max_value=15,
            value=3,
            step=1,
            help="Number of times the user opened or unlocked into the app.",
            key="input_launches"
        )

        interactions_choice = st.number_input(
            "User In-App Interactions (Taps, Swipes, Clicks)",
            min_value=0,
            max_value=30,
            value=6,
            step=1,
            help="Level of in-app activity during the session.",
            key="input_interactions"
        )

    # Predict Button
    predict_clicked = st.button("🚀 Analyze & Predict Distraction Pattern", type="primary", use_container_width=True)

    if predict_clicked or "prediction_run" in st.session_state:
        st.session_state["prediction_run"] = True

        if pipeline is None:
            st.error("⚠️ Model pipeline file `models/model.joblib` not found. Please run `python src/train_model.py` first.")
        else:
            # Build input DataFrame
            input_df = pd.DataFrame([{
                "category": category_choice,
                "screen_time_min": float(screen_time_choice),
                "launches": int(launches_choice),
                "interactions": int(interactions_choice),
                "hour": int(hour_choice),
                "day_of_week": day_choice
            }])

            # Inference
            prediction = pipeline.predict(input_df)[0]
            classes = pipeline.classes_
            probs = pipeline.predict_proba(input_df)[0]
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
            confidence = prob_dict[prediction] * 100

            # UI Cards
            if prediction == "High Distraction":
                card_class = "result-high"
                badge_bg = "#ef4444"
                icon = "🚨"
                tagline = "High Distraction Episode Detected"
                advice = (
                    "**Actionable Advice:** Prolonged engagement in non-work apps can erode focus. "
                    "Consider setting an app limit (e.g. 25 min), silencing non-essential notifications, "
                    "and winding down devices 1 hour before sleep."
                )
            elif prediction == "Moderate Distraction":
                card_class = "result-mod"
                badge_bg = "#f59e0b"
                icon = "⚖️"
                tagline = "Moderate Engagement Pattern"
                advice = (
                    "**Actionable Advice:** Your usage is currently manageable. Ensure it does not "
                    "gradually expand into an extended distraction session by taking periodic breaks."
                )
            else:
                card_class = "result-low"
                badge_bg = "#10b981"
                icon = "✅"
                tagline = "Low Distraction / Focused Session"
                advice = (
                    "**Actionable Advice:** Excellent digital hygiene! This session was either brief or directed "
                    "toward functional tasks. Keep up this balanced rhythm."
                )

            st.markdown(f"""
            <div class="result-card {card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span class="badge-pill" style="background-color: {badge_bg}; color: white;">{icon} {prediction}</span>
                    <span style="font-size: 1.1rem; font-weight: 600;">Model Confidence: {confidence:.1f}%</span>
                </div>
                <h3 style="margin: 0.5rem 0 0.8rem 0; color: white;">{tagline}</h3>
                <p style="font-size: 1rem; line-height: 1.6; color: #f1f5f9; margin-bottom: 0.5rem;">
                    Based on <strong>{screen_time_choice:.0f} minutes</strong> of <strong>{category_choice}</strong> app usage
                    with <strong>{launches_choice} launches</strong> and <strong>{interactions_choice} interactions</strong> at <strong>{hour_choice:02d}:00 on {day_choice}</strong>.
                </p>
                <div style="margin-top: 1rem; padding: 0.8rem 1rem; background: rgba(0,0,0,0.25); border-radius: 8px;">
                    {advice}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Class Probability Distribution
            st.markdown("#### 📊 Model Probability Distribution")
            p_cols = st.columns(3)
            order_classes = ["Low Distraction", "Moderate Distraction", "High Distraction"]
            colors = {"Low Distraction": "#10b981", "Moderate Distraction": "#f59e0b", "High Distraction": "#ef4444"}

            for idx, c_name in enumerate(order_classes):
                with p_cols[idx]:
                    val = prob_dict.get(c_name, 0.0) * 100
                    st.metric(label=c_name, value=f"{val:.1f}%")
                    st.progress(val / 100.0)


# =========================================================
# TAB 2: DATASET & ANALYTICS
# =========================================================
with tab_analytics:
    st.subheader("Exploratory Data Analysis Dashboard")
    st.markdown("Explore real usage metrics across 3,000 recorded app sessions from 101 unique users.")

    if df is not None:
        # Top KPI cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">3,000</div>
                <div class="metric-label">Logged Sessions</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">101</div>
                <div class="metric-label">Tracked Users</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df['screen_time_min'].median():.1f}m</div>
                <div class="metric-label">Median Screen Time</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">20 Apps</div>
                <div class="metric-label">4 Key Categories</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Visualizations
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### ⏱️ Screen Time by App Category")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            cat_palette = {"Social": "#9b59b6", "Entertainment": "#e67e22", "Productivity": "#27ae60", "Utilities": "#3498db"}
            sns.boxplot(x="category", y="screen_time_min", hue="category", data=df, palette=cat_palette, ax=ax1, legend=False)
            ax1.set_xlabel("App Category")
            ax1.set_ylabel("Screen Time (min)")
            ax1.set_title("Session Duration Distribution", fontweight="bold")
            st.pyplot(fig1)

        with chart_col2:
            st.markdown("##### 🌙 24-Hour Diurnal Screen Time Pattern")
            hourly = df.groupby("hour")["screen_time_min"].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(hourly["hour"], hourly["screen_time_min"], marker="o", color="#3b82f6", linewidth=2)
            ax2.axvspan(22, 24, alpha=0.15, color="red", label="Late Night")
            ax2.axvspan(0, 5, alpha=0.15, color="red")
            ax2.set_xlabel("Hour of Day (0-23)")
            ax2.set_ylabel("Avg Screen Time (min)")
            ax2.set_title("Mean Screen Time Throughout 24h Day", fontweight="bold")
            ax2.legend()
            st.pyplot(fig2)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.markdown("##### 👆 App Launches vs. In-App Interactions")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                x="launches",
                y="interactions",
                hue="category",
                data=df.sample(min(800, len(df)), random_state=42),
                palette=cat_palette,
                alpha=0.6,
                ax=ax3
            )
            ax3.set_xlabel("Launches Count")
            ax3.set_ylabel("Interactions (Taps/Clicks)")
            ax3.set_title("Compulsive Check Correlation", fontweight="bold")
            st.pyplot(fig3)

        with chart_col4:
            st.markdown("##### 📊 App Distribution in Dataset")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            cat_counts = df["category"].value_counts()
            ax4.pie(
                cat_counts,
                labels=cat_counts.index,
                autopct="%1.1f%%",
                colors=["#9b59b6", "#3498db", "#27ae60", "#e67e22"],
                startangle=140
            )
            ax4.set_title("Session Share Across Categories", fontweight="bold")
            st.pyplot(fig4)

        # Expandable Data Table
        with st.expander("🔍 Inspect Raw Dataset Records (Filterable)", expanded=False):
            cat_filter = st.multiselect(
                "Filter by Category:",
                options=df["category"].unique(),
                default=df["category"].unique()
            )
            filtered_df = df[df["category"].isin(cat_filter)][
                ["user_id", "date", "app_name", "category", "screen_time_min", "launches", "interactions", "is_productive"]
            ]
            st.dataframe(filtered_df.head(100), use_container_width=True)
    else:
        st.warning("Dataset not found. Please ensure `screen_time_app_usage_dataset.csv` exists in `data/`.")


# =========================================================
# TAB 3: MODEL PERFORMANCE & VIVA SHOWCASE
# =========================================================
with tab_model:
    st.subheader("Model Evaluation, Comparison & Viva Showcase")
    st.markdown(
        "Demonstrate the machine learning rigour of the project. "
        "We compared a **Multinomial Logistic Regression baseline** with an ensemble **Random Forest Classifier**."
    )

    if metrics:
        rf_meta = metrics["models"]["Random Forest Classifier"]
        base_meta = metrics["models"]["Baseline (Logistic Regression)"]

        # Comparison cards
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #64748b;">
                <div class="metric-label">Baseline Model</div>
                <h3 style="margin: 0.3rem 0; color: #f1f5f9;">Logistic Regression</h3>
                <div class="metric-value" style="color: #94a3b8;">{base_meta['test_accuracy']*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">Linear decision boundaries, rapid training baseline.</p>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #38bdf8;">
                <div class="metric-label">Primary Champion Model</div>
                <h3 style="margin: 0.3rem 0; color: #f1f5f9;">Random Forest Classifier</h3>
                <div class="metric-value" style="color: #38bdf8;">{rf_meta['test_accuracy']*100:.2f}%</div>
                <div class="metric-label">Test Accuracy</div>
                <p style="font-size: 0.85rem; color: #38bdf8; margin-top: 0.5rem;">Ensemble of 120 decision trees capturing non-linear interactions.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Evaluation KPI metrics
        st.markdown("#### 🎯 Random Forest Test Set Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{rf_meta['test_accuracy']*100:.2f}%")
        k2.metric("Precision (Weighted)", f"{rf_meta['test_precision']*100:.2f}%")
        k3.metric("Recall (Weighted)", f"{rf_meta['test_recall']*100:.2f}%")
        k4.metric("F1-Score (Weighted)", f"{rf_meta['test_f1']*100:.2f}%")

        # Visualizations (Confusion Matrix & Feature Importances)
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("##### 🔲 Confusion Matrix (Test Set)")
            cm_img_path = "notebooks_or_reports/figures/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Confusion matrix image will appear after running `python src/train_model.py`.")

        with fig_col2:
            st.markdown("##### 🌟 Top Feature Importances")
            fi_img_path = "notebooks_or_reports/figures/feature_importances.png"
            if os.path.exists(fi_img_path):
                st.image(fi_img_path, use_container_width=True)
            else:
                st.info("Feature importance image will appear after running `python src/train_model.py`.")

        # Viva Guide Accordions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🎓 College Viva & Presentation Q&A")

        with st.expander("Q1: Why did you not just use deep learning / neural networks?"):
            st.markdown("""
            **Answer:**
            For tabular datasets with 3,000 records and structured continuous/categorical features, ensemble tree algorithms like **Random Forest** consistently outperform deep neural networks in empirical benchmarks, require far less computational overhead, avoid overfitting, and provide transparent Gini feature importance.
            """)

        with st.expander("Q2: How was data leakage prevented?"):
            st.markdown("""
            **Answer:**
            1. We strictly separated the dataset into 80% train and 20% test before calculating any scaling statistics.
            2. We used `StratifiedShuffleSplit` logic (`stratify=y`) to maintain identical class ratios across both splits.
            3. All preprocessing transformers (`StandardScaler`, `OneHotEncoder`) were wrapped in Scikit-learn's `Pipeline` so test samples were only transformed using training-derived parameters.
            """)

        with st.expander("Q3: Why couldn't `is_productive` be used directly as the target?"):
            st.markdown("""
            **Answer:**
            Inspecting the raw dataset revealed that `is_productive` is literally identical to `category == 'Productivity'` (753 rows each). If the input includes app category or app name, predicting `is_productive` is a trivial tautology (100% leak) rather than a realistic distraction predictor.
            """)
    else:
        st.info("Metrics not found. Please train models by running `python src/train_model.py`.")


# =========================================================
# TAB 4: BEHAVIORAL ARCHETYPES (UNSUPERVISED)
# =========================================================
with tab_clusters:
    st.subheader("Unsupervised Behavioral Clustering (K-Means)")
    st.markdown(
        "To explore usage habits without relying on rule-based categories, we applied **K-Means Clustering ($K=3$)** "
        "directly on continuous usage dimensions (`screen_time_min`, `launches`, `interactions`, `hour`)."
    )

    if metrics and "cluster_profiles" in metrics:
        profiles = metrics["cluster_profiles"]
        archetype_names = {
            "0": ("Focused / Low-Engagement Utility", "Quick functional check-ins with low screen time (~20 min) and standard interaction counts.", "#10b981"),
            "1": ("Prolonged / High-Intensity Binge", "Extended continuous screen time (~93 min), representing long streaming, gaming, or social rabbit holes.", "#ef4444"),
            "2": ("Early Morning / Transition Checks", "Brief sessions (~20 min) occurring in the early morning or transition periods.", "#3b82f6")
        }

        cols = st.columns(3)
        for idx, (cid, pdata) in enumerate(profiles.items()):
            name, desc, color = archetype_names.get(cid, (f"Cluster {cid}", "Usage pattern", "#64748b"))
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {color}; text-align: left;">
                    <h4 style="color: {color}; margin-top: 0;">Archetype {chr(65+int(cid))}: {name}</h4>
                    <p style="font-size: 0.9rem; color: #cbd5e1; min-height: 55px;">{desc}</p>
                    <hr style="border-color: rgba(255,255,255,0.1);">
                    <div style="font-size: 0.85rem; color: #94a3b8;">
                        • <strong>Mean Screen Time:</strong> {pdata['screen_time_min']:.1f} min<br>
                        • <strong>Mean Launches:</strong> {pdata['launches']:.1f}<br>
                        • <strong>Mean Interactions:</strong> {pdata['interactions']:.1f}<br>
                        • <strong>Avg Hour of Day:</strong> {int(pdata['hour'])}:00
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        > **Academic Takeaway:** The unsupervised clustering independently proves that **session duration** and **session timing** are the primary axes that differentiate healthy digital habits from binge behaviors, confirming our supervised feature importance findings.
        """)
    else:
        st.info("Cluster profiles not available. Run `python src/train_model.py` to generate them.")


# =========================================================
# TAB 5: PROJECT & ETHICAL AI
# =========================================================
with tab_about:
    st.subheader("About the Project & Ethical Disclaimers")

    st.markdown("""
    ### 🎯 Project Purpose
    The **Digital Distraction Pattern Predictor** was developed to bridge digital wellbeing research with practical machine learning. It provides an explainable framework for students, developers, and educators to understand:
    1. How digital usage sessions translate into distraction risk profiles.
    2. How to clean tabular behavioral data, detect synthetic noise columns, and prevent data leakage.
    3. How to train, evaluate, and interpret supervised and unsupervised Scikit-learn pipelines.

    ---

    ### ⚠️ Ethical AI & Clinical Scope Disclaimer
    - **Not a Diagnostic Tool:** This application is strictly an exploratory educational and digital wellbeing tool. It **does NOT** diagnose medical, psychological, or psychiatric conditions such as Attention Deficit Hyperactivity Disorder (ADHD), clinical depression, or internet addiction disorder.
    - **Dataset Limitations:** The dataset is derived from 3,000 logged session records from Kaggle. In real-world settings, digital distraction depends on individual user goals, context, and intent (e.g. reading educational articles on social media vs. mindless doomscrolling).
    - **No Fabrication:** The ML pipeline strictly avoids fabricating synthetic user records or pretending heuristic labels are clinical ground truth.

    ---

    ### 🛠️ Technology Stack
    - **Core:** Python 3.12, Pandas, NumPy
    - **Machine Learning:** Scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomForestClassifier`, `KMeans`)
    - **Persistence:** Joblib
    - **Web Framework:** Streamlit
    - **Visualization:** Matplotlib, Seaborn
    """)

    st.info("Project developed for College Machine Learning Demonstration & Viva Presentation.")
