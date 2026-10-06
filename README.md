---
title: Explainable AI Customer Churn Predictor
emoji: 📡
colorFrom: blue
colorTo: orange
sdk: streamlit
sdk_version: "1.64.0"
app_file: app.py
pinned: false
license: mit
---

# Explainable AI Customer Churn Predictor

**Predict customer churn risk and understand the factors behind each prediction.**

A production-style Streamlit application that predicts telecom customer churn and explains every prediction with SHAP — built so a non-technical retention team can use it, not just a data scientist.

## What it does

1. Predicts the probability a customer will churn
2. Shows a Low / Medium / High / Critical risk category
3. Explains the prediction with SHAP (factors increasing risk, factors reducing risk)
4. Suggests business-oriented retention actions based on the top risk drivers
5. Shows global model explainability (what the model relies on overall)
6. Reports real model performance on a held-out test set

## Try it

Run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Deploy to Hugging Face Spaces: push this repo to a new Space with the **Streamlit** SDK — the YAML block at the top of this README configures it automatically.

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — ~7,043 customers. A copy is included at `data/Telco-Customer-Churn.csv` so the pipeline is reproducible; if you replace it, keep the same column names.

## Pipeline

```
Raw Dataset → Data Cleaning → Feature Engineering → Train/Test Split
   → Preprocessing → Class Imbalance Handling (SMOTE, train-only)
   → Model Comparison → Bayesian Hyperparameter Optimization (Optuna)
   → Probability Calibration (isotonic) → Threshold Optimization
   → SHAP Explainability → Saved Model Artifacts → Streamlit App
```

Run once, offline:
```bash
python train_pipeline.py
```
This writes everything the app needs into `models/`. **The app never retrains — it only loads these saved artifacts.**

### Data cleaning

- `TotalCharges` arrives as text with 11 blank values, all belonging to brand-new customers with `tenure = 0` (never billed yet). Converted to numeric and filled with `0`, not the column mean — these are legitimately zero, not missing-at-random.
- Checked for duplicate `customerID` rows (none found).
- No data leakage: the test split is created before any preprocessing, resampling, or feature-selection step is fit.

### Engineered features

| Feature | What it is |
|---|---|
| `tenure_group` | Coarse tenure bucket (0-12, 13-24, 25-48, 49-72 months) — captures the early-tenure churn spike more explicitly than a raw continuous value. |
| `num_services` | Count of the 6 optional add-on services subscribed to (0-6) — a proxy for account "stickiness". |
| `avg_monthly_spend` | `TotalCharges / tenure` — a smoothed spend rate, distinct from the customer's *current* `MonthlyCharges`. |
| `has_internet_addons` | 1 if the customer has at least one add-on service, else 0. |

### Model comparison (5-fold CV, training data)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| LightGBM | 0.9365 | 0.9448 | 0.8650 | 0.8439 | 0.8543 | 0.8561 |
| Random Forest | 0.9338 | 0.9259 | 0.8612 | 0.8712 | 0.8662 | 0.8654 |
| XGBoost | 0.9314 | 0.9404 | 0.8557 | 0.8425 | 0.8490 | 0.8502 |
| Logistic Regression | 0.8522 | 0.8378 | 0.7518 | 0.8050 | 0.7775 | 0.7696 |

LightGBM edges out the other tree-based models on baseline ROC-AUC, but the margin over XGBoost (0.005) is within noise. **XGBoost was selected as the tuning target** for its more mature tooling for isotonic calibration and stable `TreeExplainer` SHAP attribution used throughout this app; LightGBM's numbers are kept above for transparency rather than hidden. Selection weighted ROC-AUC, PR-AUC, and **recall** more heavily than raw accuracy, since for churn prediction a missed churner (false negative) is usually costlier than a false retention alarm.

### Hyperparameter optimization

Optuna, 30 trials, Tree-structured Parzen Estimator sampler, optimizing 5-fold CV ROC-AUC on the SMOTE-balanced training data. Best CV ROC-AUC: **0.9409**.

### Calibration & threshold

- Final model probabilities calibrated with **isotonic regression** (5-fold), train Brier score **0.0986**.
- Decision threshold optimized by maximizing F1 on the real (non-resampled) training class balance — **not** the default 0.50. Selected threshold: **`CHURN_THRESHOLD = 0.35`**.

### Held-out test set performance (1,409 customers, never used in training or tuning)

| Metric | Value |
|---|---|
| Accuracy | 0.7686 |
| Precision | 0.5504 |
| Recall | 0.7005 |
| F1 | 0.6165 |
| ROC-AUC | 0.8392 |
| PR-AUC | 0.6424 |
| Brier score | 0.1417 |

### Risk category boundaries

| Category | Probability range |
|---|---|
| Low | 0% – 25% |
| Medium | 25% – 50% |
| High | 50% – 75% |
| Critical | 75% – 100% |

Chosen as even quartile bands over the predicted-probability range rather than arbitrary round numbers; documented here rather than silently hard-coded in the app.

## SHAP explainability

Every prediction gets an individual SHAP explanation (top risk-increasing and risk-reducing factors, as a diverging bar chart), not just a global importance chart. The global "How the model decides" tab shows the top 10 features by mean |SHAP value| on the held-out test set, computed once and cached — it does not recompute at request time.

**Top global drivers of churn:** month-to-month contract (by a wide margin), short tenure, no online security, no tech support, electronic check payment, two-year contract (protective), fiber optic internet, spend rate, monthly charges, paperless billing.

## Repository structure

```
churn-predictor/
├── app.py                 # Streamlit app — loads artifacts, never retrains
├── train_pipeline.py       # Full offline training pipeline (run once)
├── requirements.txt
├── data/
│   └── Telco-Customer-Churn.csv
├── models/                 # Saved artifacts produced by train_pipeline.py
│   ├── churn_model.joblib      # calibrated final model
│   ├── shap_model.joblib       # uncalibrated tuned model, used for SHAP
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── shap_background.joblib
│   ├── metadata.joblib
│   └── metadata.json
└── docs/
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · SHAP · Optuna · imbalanced-learn (SMOTE) · Streamlit · Plotly · joblib

## Limitations & honesty note

Published benchmarks on this exact public dataset generally top out around 80-82% accuracy / 0.85-0.87 ROC-AUC — a meaningful share of churners are statistically indistinguishable from retained customers on the available features. The numbers above are the actual held-out test results, with no data leakage or test-set tuning. This is a portfolio/demonstration project on a public benchmark dataset, not a validated production system — treat its outputs as a statistical estimate, not certainty about any individual customer.

## License

MIT — see [LICENSE](LICENSE).
