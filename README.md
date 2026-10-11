# 📱 Digital Distraction Pattern Predictor

A beginner-friendly Machine Learning project that analyzes smartphone application usage habits and predicts digital distraction behavior patterns using Scikit-Learn and Streamlit.

---

## 📌 Project Overview

With the pervasive use of smartphones, distinguishing between intentional digital utility and unproductive digital distraction has become a critical challenge in personal productivity and digital wellbeing.

The **Digital Distraction Pattern Predictor**:
1. Ingests smartphone screen time and session logs (`screen_time_app_usage_dataset.csv`).
2. Cleans raw session data, filters synthetic benchmark noise, and extracts temporal usage patterns.
3. Formulates a transparent, domain-grounded **Digital Distraction Index** (Low, Moderate, High Distraction).
4. Trains and compares a **Multinomial Logistic Regression baseline** against an ensemble **Random Forest Classifier** achieving **91.5% accuracy** on unseen test data.
5. Employs unsupervised **K-Means Clustering** to discover natural digital usage archetypes.
6. Delivers an interactive **Streamlit web application** featuring live prediction, analytics dashboards, model evaluation metrics, and practical digital wellbeing advice.

---

## 📂 Project Structure

```
Digital Distraction Pattern Predictor/
├── data/
│   └── screen_time_app_usage_dataset.csv     # 3,000 real session logs (preserved)
├── models/
│   ├── model.joblib                         # Trained Random Forest pipeline
│   ├── kmeans_model.joblib                  # Trained K-Means clustering model
│   └── model_metrics.json                   # Exported evaluation metrics & importances
├── notebooks_or_reports/
│   ├── eda_report.md                        # Comprehensive analysis & viva report
│   └── figures/                             # Generated analytical plots
│       ├── distribution_screen_time.png
│       ├── category_screen_time.png
│       ├── distraction_level_distribution.png
│       ├── launches_vs_interactions.png
│       ├── usage_by_hour.png
│       ├── confusion_matrix.png
│       └── feature_importances.png
├── src/
│   ├── data_analysis.py                     # Exploratory Data Analysis & plot generator
│   ├── train_model.py                       # ML pipeline training & evaluation
│   └── predict.py                           # CLI inference with validation & recommendations
├── app.py                                   # Interactive Streamlit Web Interface
├── requirements.txt                         # Exact dependencies
├── README.md                                # Project documentation & viva guide
└── .gitignore                               # Environment & cache ignore rules
```

---

## 🔍 Dataset Findings & Honest Domain Analysis

### 1. Dataset Dimensions & Coverage
- **3,000 session records** across **101 unique users** (`user_id`: 1000 to 1100).
- Recorded over a 3-month window from **January 1, 2024 to April 1, 2024**.
- 20 popular applications across 4 categories: `Social`, `Entertainment`, `Productivity`, and `Utilities`.

### 2. Detection of Data Peculiarities
- **Absence of Ground-Truth Distraction Label:** The raw dataset does not contain an explicit clinical distraction label.
- **`is_productive` Tautology:** In the raw dataset, `is_productive` is simply a boolean flag equal to `category == 'Productivity'`. Predicting `is_productive` when app category or app name is known is trivial data leakage.
- **Sparse YouTube Metrics:** `youtube_views`, `youtube_likes`, and `youtube_comments` are 95.3% null (populated only for the YouTube app) and are not generalizable across other applications.
- **Synthetic Noise Columns:** Columns `extra_col_11` through `extra_col_23` contain Gaussian $\mathcal{N}(50, 10^2)$ noise without domain relevance and were discarded to prevent overfitting.

### 3. Transparent Domain-Justified Distraction Index
To avoid fabricating synthetic data, we formulated a transparent behavioral index grounded in digital wellbeing research:

$$\text{Distraction Index} = 1.5 \times W_{\text{category}} + 1.2 \times S_{\text{time}} + 0.8 \times C_{\text{compulsion}} + 0.3 \times P_{\text{night}}$$

- **Category Weights ($W_{\text{category}}$):** `Social` (1.0), `Entertainment` (0.9), `Utilities` (0.3), `Productivity` (0.0).
- **Duration Factor ($S_{\text{time}}$):** $\min(\text{screen\_time\_min} / 60, 1.5)$.
- **Compulsion Factor ($C_{\text{compulsion}}$):** $0.5 \times (\text{launches}/4) + 0.5 \times (\text{interactions}/8)$.
- **Circadian Disruption Penalty ($P_{\text{night}}$):** $+0.3$ if session started between 22:00 and 05:00.

**Categorization:**
- **Low Distraction ($\le 1.49$):** 990 samples (33.0%) — brief sessions, productivity/utility tasks.
- **Moderate Distraction ($1.49 - 2.27$):** 990 samples (33.0%) — balanced daytime engagement.
- **High Distraction ($> 2.27$):** 1,020 samples (34.0%) — prolonged streaming/social loops or late-night binging.

> **Academic Disclaimer:** This is an exploratory behavioral heuristic. It is not a clinical diagnosis of ADHD or internet addiction disorder.

---

## 📊 Machine Learning Model Performance

We trained models using an 80/20 stratified split (2,400 training samples, 600 test samples). All preprocessing (`StandardScaler` and `OneHotEncoder`) was wrapped inside Scikit-learn's `Pipeline` to prevent data leakage.

| Metric | Baseline: Logistic Regression | Champion: Random Forest Classifier |
| :--- | :--- | :--- |
| **Train Accuracy** | 92.79% | 99.21% |
| **Test Accuracy** | 89.17% | **91.50%** |
| **Test Precision (Weighted)** | 89.30% | **91.57%** |
| **Test Recall (Weighted)** | 89.17% | **91.50%** |
| **Test F1-Score (Weighted)** | 89.21% | **91.53%** |

### Top Predictive Features (Gini Importance):
1. `screen_time_min` (Continuous session duration)
2. `category_Social` (Presence of social media platform)
3. `category_Utilities` & `category_Productivity` (Functional context)
4. `interactions` (Touch frequency)
5. `launches` (Compulsive opens)
6. `hour` (Diurnal timing)

---

## ⚡ Quick Start Guide (PowerShell Commands)

### 1. Prerequisites & Environment Activation
Open your terminal in the project directory in VS Code and activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

*(If prompted about script execution policy, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 2. Verify / Install Dependencies
```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Run Exploratory Data Analysis (EDA)
Inspects the raw CSV, prints summary statistics, and saves charts to `notebooks_or_reports/figures/`:
```powershell
& .\.venv\Scripts\python.exe src/data_analysis.py
```

### 4. Train and Evaluate Machine Learning Pipeline
Trains the baseline and Random Forest models, performs K-Means clustering, evaluates test metrics, and exports `models/model.joblib`:
```powershell
& .\.venv\Scripts\python.exe src/train_model.py
```

### 5. Run CLI Prediction
Test the trained pipeline from the terminal with sample inputs:
```powershell
# High distraction scenario (Late night social binging)
& .\.venv\Scripts\python.exe src/predict.py --category Social --screen_time 75 --launches 4 --interactions 8 --hour 23 --day Saturday

# Low distraction scenario (Daytime work session)
& .\.venv\Scripts\python.exe src/predict.py --category Productivity --screen_time 20 --launches 1 --interactions 3 --hour 14 --day Wednesday
```

### 6. Launch the Streamlit Web Application
```powershell
& .\.venv\Scripts\streamlit.exe run app.py
```
Open your web browser at: **`http://localhost:8501`**

---

## 🎓 College Viva & Presentation Preparation

Be prepared to answer these common viva questions with confidence:

### 1. What problem does this project solve?
> "It analyzes smartphone session metrics (screen time, launch frequency, interactions, app categories, and time of day) to classify digital usage into Low, Moderate, or High Distraction patterns and offers actionable digital wellbeing recommendations."

### 2. Why didn't you predict the `is_productive` column that came in the dataset?
> "During exploratory data analysis, we discovered that `is_productive` is literally identical to `category == 'Productivity'`. If our model knows the app category, predicting `is_productive` would be 100% data leakage rather than real machine learning. We instead built a composite distraction pattern model that accounts for duration, compulsion, and late-night habits."

### 3. How did you prevent data leakage?
> "We split the data before applying transformations using an 80/20 stratified split (`stratify=y`) so class proportions matched in both splits. Preprocessing (`StandardScaler` for numerical metrics and `OneHotEncoder` for categories) was encapsulated inside Scikit-learn's `Pipeline`, ensuring the test set was never seen during scaling or fitting."

### 4. Why Random Forest instead of a Deep Neural Network?
> "Tabular datasets with structured features perform exceptionally well with ensemble decision trees. Random Forest avoids the extensive hyperparameter search and computational cost of deep learning, resists overfitting through bagging, handles non-linear feature interactions, and provides interpretable feature importances."

### 5. How did you evaluate the model?
> "We evaluated the model on an untouched 20% test set (600 samples) using Test Accuracy (91.5%), Precision (91.6%), Recall (91.5%), F1-Score (91.5%), and a Confusion Matrix. We compared it against a Logistic Regression baseline (89.2% accuracy) to prove the value of non-linear tree ensembles."

---

## 📋 Deliverables Verification

- [x] Working ML training pipeline (`src/train_model.py`)
- [x] Saved Joblib models (`models/model.joblib` and `models/kmeans_model.joblib`)
- [x] Streamlit web application (`app.py`) running at `http://localhost:8501`
- [x] Dataset analysis and visualizations in `notebooks_or_reports/figures/`
- [x] Detailed EDA report (`notebooks_or_reports/eda_report.md`)
- [x] Complete README with exact PowerShell commands and viva guide
