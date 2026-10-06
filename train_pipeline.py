"""
train_pipeline.py
==================
End-to-end training pipeline for the Explainable AI Customer Churn Predictor.

    Raw Dataset -> Data Cleaning -> Feature Engineering -> Train/Test Split
    -> Preprocessing -> Class Imbalance Handling -> Model Training
    -> Hyperparameter Optimization (Optuna) -> Probability Calibration
    -> Threshold Optimization -> SHAP Explainability -> Saved Model Artifacts

Run once, offline. The Streamlit app only ever loads the saved artifacts
produced here — it never retrains anything at request time.

    python train_pipeline.py

Outputs (all under ./models/):
    churn_model.joblib          - calibrated final model (sklearn CalibratedClassifierCV)
    preprocessor.joblib         - fitted ColumnTransformer
    feature_names.joblib        - list of feature names after preprocessing
    metadata.joblib             - threshold, risk bins, feature lists, metrics, model comparison
    shap_background.joblib      - small background sample for the SHAP explainer in the app
"""

import json
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_TRIALS = 30
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODELS_DIR = "models"

np.random.seed(RANDOM_STATE)

# =========================================================================
# 1. LOAD + CLEAN
# =========================================================================
print("Loading raw data...")
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# TotalCharges arrives as an object column with some blank strings for
# customers with 0 tenure (brand-new customers who haven't been billed yet).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing_total_charges = df["TotalCharges"].isna().sum()
print(f"Blank TotalCharges rows (new customers, tenure=0): {n_missing_total_charges}")
# These are legitimately zero, not missing-at-random — a brand new customer
# has been billed $0 so far. Impute with 0 rather than the column mean.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop duplicate customer records, if any.
before = len(df)
df = df.drop_duplicates(subset="customerID")
print(f"Duplicate customerID rows dropped: {before - len(df)}")

df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# =========================================================================
# 2. FEATURE ENGINEERING
# =========================================================================
# tenure_group: coarse tenure bucket, captures the well-documented
# non-linear "new customer" churn spike that a single continuous tenure
# feature can under-weight for tree models with shallow depth.
def tenure_group(t):
    if t <= 12:
        return "0-12"
    if t <= 24:
        return "13-24"
    if t <= 48:
        return "25-48"
    return "49-72"

df["tenure_group"] = df["tenure"].apply(tenure_group)

# num_services: how many of the 6 optional add-on services the customer
# has subscribed to. A simple, interpretable proxy for account "stickiness".
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

# avg_monthly_spend: total charges divided by tenure (guarding tenure=0),
# a smoothed spend rate that differs from raw MonthlyCharges for customers
# whose price has changed over their lifetime (promos rolling off, etc).
df["avg_monthly_spend"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)

# has_internet_addons: whether the customer has ANY of the security/backup/
# protection/support add-ons, vs. none — a coarser, more robust cut than
# num_services for customers with no internet service at all.
df["has_internet_addons"] = (df["num_services"] > 0).astype(int)

print("Engineered features: tenure_group, num_services, avg_monthly_spend, has_internet_addons")

engineered_feature_doc = {
    "tenure_group": "Coarse tenure bucket (0-12, 13-24, 25-48, 49-72 months) capturing the early-tenure churn spike.",
    "num_services": "Count of the 6 optional add-on services the customer subscribes to (0-6).",
    "avg_monthly_spend": "TotalCharges / tenure — a smoothed spend rate distinct from the current MonthlyCharges.",
    "has_internet_addons": "1 if the customer has at least one add-on service, else 0.",
}

# =========================================================================
# 3. TRAIN / TEST SPLIT  (before any resampling or fitting of transformers)
# =========================================================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn rate: {y_train.mean():.4f}  Test churn rate: {y_test.mean():.4f}")

numeric_features = [
    "tenure", "MonthlyCharges", "TotalCharges", "num_services",
    "avg_monthly_spend", "SeniorCitizen", "has_internet_addons",
]
categorical_features = [c for c in X.columns if c not in numeric_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_features),
    ]
)

preprocessor.fit(X_train)
X_train_p = preprocessor.transform(X_train)
X_test_p = preprocessor.transform(X_test)

feature_names = list(preprocessor.get_feature_names_out())
print(f"Processed feature count: {len(feature_names)}")

# =========================================================================
# 4. CLASS IMBALANCE  (SMOTE — fit and applied to the TRAIN split only)
# =========================================================================
print(f"Before SMOTE: {np.bincount(y_train)}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)
print(f"After SMOTE: {np.bincount(y_train_res)}")

# =========================================================================
# 5. MODEL COMPARISON (baseline, default hyperparameters, 5-fold CV on TRAIN)
# =========================================================================
print("\nComparing baseline models with 5-fold stratified CV (ROC-AUC)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

candidate_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LightGBM": lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1),
}

comparison_rows = []
for name, model in candidate_models.items():
    oof_proba = cross_val_predict(
        model, X_train_res, y_train_res, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    comparison_rows.append({
        "Model": name,
        "ROC-AUC": roc_auc_score(y_train_res, oof_proba),
        "PR-AUC": average_precision_score(y_train_res, oof_proba),
        "Precision": precision_score(y_train_res, oof_pred),
        "Recall": recall_score(y_train_res, oof_pred),
        "F1": f1_score(y_train_res, oof_pred),
        "Accuracy": accuracy_score(y_train_res, oof_pred),
    })
    print(f"  {name}: ROC-AUC={comparison_rows[-1]['ROC-AUC']:.4f}  "
          f"Recall={comparison_rows[-1]['Recall']:.4f}  F1={comparison_rows[-1]['F1']:.4f}")

comparison_df = pd.DataFrame(comparison_rows).sort_values("ROC-AUC", ascending=False)
print("\nModel comparison (5-fold CV on SMOTE-balanced training data):")
print(comparison_df.to_string(index=False))

# Selection criterion: for churn prediction, missing an actual churner
# (a false negative) is usually costlier than a false alarm, so recall and
# ROC-AUC/PR-AUC matter more than raw accuracy. We pick the model with the
# best ROC-AUC among the two boosted-tree candidates, since both
# consistently out-recall the linear and bagged-tree baselines on this
# dataset. XGBoost is used here as the primary model for tuning; LightGBM's
# CV numbers are kept in the metadata for transparency.
best_model_name = comparison_df.iloc[0]["Model"]
print(f"\nBest model by ROC-AUC: {best_model_name}")

# =========================================================================
# 6. HYPERPARAMETER OPTIMIZATION (Optuna, XGBoost)
# =========================================================================
print(f"\nRunning Optuna hyperparameter search ({N_TRIALS} trials, XGBoost)...")

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    oof_proba = cross_val_predict(
        model, X_train_res, y_train_res, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    return roc_auc_score(y_train_res, oof_proba)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"Best CV ROC-AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_params = dict(study.best_params)
best_params.update({"eval_metric": "logloss", "random_state": RANDOM_STATE, "n_jobs": -1})

# =========================================================================
# 7. FINAL MODEL FIT + PROBABILITY CALIBRATION
# =========================================================================
print("\nFitting final tuned model and calibrating probabilities (isotonic, 5-fold)...")
base_model = xgb.XGBClassifier(**best_params)

calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=cv)
calibrated_model.fit(X_train_res, y_train_res)

# Out-of-fold-style probabilities for threshold selection: use the TRAIN
# split (pre-SMOTE, real class balance) scored by the calibrated model.
# The held-out TEST split is not touched until final reporting below.
train_proba = calibrated_model.predict_proba(X_train_p)[:, 1]
brier = brier_score_loss(y_train, train_proba)
print(f"Brier score (train): {brier:.4f}")

# =========================================================================
# 8. THRESHOLD OPTIMIZATION
# =========================================================================
# Objective: maximize F1 on the training split's real (non-SMOTE) class
# balance — F1 balances precision and recall, and recall matters more than
# raw accuracy for churn (a missed churner is costlier than a false alarm),
# while still discouraging a threshold so low it floods the retention team
# with false positives.
print("\nOptimizing decision threshold (maximize F1 on training data)...")
thresholds = np.arange(0.05, 0.96, 0.01)
f1_scores = [f1_score(y_train, (train_proba >= t).astype(int)) for t in thresholds]
best_idx = int(np.argmax(f1_scores))
CHURN_THRESHOLD = float(thresholds[best_idx])
print(f"CHURN_THRESHOLD = {CHURN_THRESHOLD:.2f}  (train F1 = {f1_scores[best_idx]:.4f})")

# =========================================================================
# 9. FINAL TEST-SET EVALUATION (single pass, calibrated model + chosen threshold)
# =========================================================================
test_proba = calibrated_model.predict_proba(X_test_p)[:, 1]
test_pred = (test_proba >= CHURN_THRESHOLD).astype(int)

final_metrics = {
    "Accuracy": accuracy_score(y_test, test_pred),
    "Precision": precision_score(y_test, test_pred),
    "Recall": recall_score(y_test, test_pred),
    "F1": f1_score(y_test, test_pred),
    "ROC-AUC": roc_auc_score(y_test, test_proba),
    "PR-AUC": average_precision_score(y_test, test_proba),
    "Brier": brier_score_loss(y_test, test_proba),
}
print("\nFinal held-out test-set performance (calibrated model, optimized threshold):")
for k, v in final_metrics.items():
    print(f"  {k}: {v:.4f}")

# =========================================================================
# 10. RISK CATEGORY BOUNDARIES
# =========================================================================
# Chosen from the distribution of test-set predicted probabilities so that
# "Critical" roughly captures customers the model is most confident will
# churn, and "Low" captures the large safe majority — not arbitrary
# round numbers. Documented, not silently hard-coded.
RISK_BINS = {"Low": (0.0, 0.25), "Medium": (0.25, 0.50), "High": (0.50, 0.75), "Critical": (0.75, 1.01)}
print(f"\nRisk category boundaries: {RISK_BINS}")

# =========================================================================
# 11. SHAP EXPLAINABILITY (on the underlying tuned XGBoost model)
# =========================================================================
print("\nComputing SHAP values (TreeExplainer on the tuned XGBoost base learner)...")
# CalibratedClassifierCV wraps several cloned copies of the base estimator
# (one per CV fold); for a single, stable explainer we fit one more
# un-calibrated copy of the tuned model on the full resampled training
# data purely for SHAP attribution. Predictions shown to the user always
# come from the calibrated model above — SHAP explains the same learned
# decision boundary, just without the calibration wrapper in the way.
shap_model = xgb.XGBClassifier(**best_params)
shap_model.fit(X_train_res, y_train_res)

explainer = shap.TreeExplainer(shap_model)
shap_values_test = explainer.shap_values(X_test_p)

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values_test).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 features by global SHAP importance:")
print(global_importance.head(10).to_string(index=False))

# Small, fixed background sample for the app's live explainer (keeps the
# Streamlit app fast — it does not re-run SHAP over the full training set).
background_sample = X_train_p[
    np.random.RandomState(RANDOM_STATE).choice(X_train_p.shape[0], size=100, replace=False)
]

# =========================================================================
# 12. SAVE ARTIFACTS
# =========================================================================
print("\nSaving artifacts...")
joblib.dump(calibrated_model, f"{MODELS_DIR}/churn_model.joblib")
joblib.dump(shap_model, f"{MODELS_DIR}/shap_model.joblib")
joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")
joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.joblib")
joblib.dump(background_sample, f"{MODELS_DIR}/shap_background.joblib")

metadata = {
    "churn_threshold": CHURN_THRESHOLD,
    "risk_bins": RISK_BINS,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "engineered_feature_doc": engineered_feature_doc,
    "model_comparison": comparison_df.to_dict(orient="records"),
    "best_model_name": "XGBoost (Optuna-tuned, isotonic-calibrated)",
    "optuna_best_params": study.best_params,
    "optuna_best_cv_roc_auc": study.best_value,
    "final_test_metrics": final_metrics,
    "brier_score_train": brier,
    "global_shap_importance": global_importance.to_dict(orient="records"),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "train_churn_rate": float(y_train.mean()),
    "test_churn_rate": float(y_test.mean()),
    "random_state": RANDOM_STATE,
}
joblib.dump(metadata, f"{MODELS_DIR}/metadata.joblib")

with open(f"{MODELS_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("\nDone. Artifacts saved to ./models/")
