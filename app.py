"""
Explainable AI Customer Churn Predictor
========================================
Streamlit application. Loads pre-trained artifacts from ./models/ (produced
by train_pipeline.py) and never retrains anything at request time.
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# -------------------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# THEME / STYLE
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0B1420; }
    section[data-testid="stSidebar"] { background-color: #101B29; border-right: 1px solid #223349; }
    h1, h2, h3, h4 { color: #E8EDF3 !important; }
    p, li, label, span { color: #C7D2DE; }
    .metric-box {
        background:#101B29; border:1px solid #223349; border-radius:8px;
        padding:18px 20px; text-align:center;
    }
    .metric-box .label { font-size:12.5px; color:#8FA1B3; margin-bottom:6px; }
    .metric-box .value { font-size:30px; font-weight:700; }
    .risk-pill {
        display:inline-block; padding:6px 16px; border-radius:20px;
        font-weight:600; font-size:15px; letter-spacing:0.02em;
    }
    .factor-card {
        background:#101B29; border:1px solid #223349; border-radius:6px;
        padding:12px 16px; margin-bottom:8px;
    }
    .disclaimer {
        font-size:12.5px; color:#8FA1B3; border-top:1px solid #223349;
        padding-top:14px; margin-top:24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {"Low": "#34D399", "Medium": "#F2C94C", "High": "#FF6B4A", "Critical": "#FF3B3B"}

# -------------------------------------------------------------------------
# LOAD ARTIFACTS (cached — loaded once per session, not per interaction)
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.joblib")
    shap_model = joblib.load("models/shap_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    metadata = joblib.load("models/metadata.joblib")
    background = joblib.load("models/shap_background.joblib")
    explainer = shap.TreeExplainer(shap_model)
    return model, shap_model, preprocessor, feature_names, metadata, background, explainer


model, shap_model, preprocessor, feature_names, metadata, background, explainer = load_artifacts()
THRESHOLD = metadata["churn_threshold"]
RISK_BINS = metadata["risk_bins"]


def risk_category(prob):
    for label, (lo, hi) in RISK_BINS.items():
        if lo <= prob < hi:
            return label
    return "Critical"


def clean_feature_label(raw_name):
    """cat__Contract_Month-to-month -> Contract: Month-to-month"""
    name = raw_name.split("__", 1)[-1]
    if "_" in name:
        parts = name.rsplit("_", 1)
        return f"{parts[0]}: {parts[1]}"
    return name


ACTION_MAP = {
    "Contract": "Offer term-based discounts or loyalty pricing to move the customer onto a longer contract.",
    "tenure": "Enroll in an early-tenure retention / onboarding program — churn risk is highest in the first year.",
    "OnlineSecurity": "Offer a free or discounted trial of the online security add-on.",
    "TechSupport": "Proactively offer tech-support enrollment.",
    "PaymentMethod": "Encourage a switch to automatic payment (bank transfer / credit card) with a small incentive.",
    "InternetService": "Review fiber-tier pricing and reliability — this segment shows elevated risk.",
    "MonthlyCharges": "Review pricing/package fit; consider a loyalty discount.",
    "avg_monthly_spend": "Review overall account value and billing history for pricing fit.",
    "TotalCharges": "Review customer lifetime value and billing history.",
    "PaperlessBilling": "Review billing experience / communication preferences.",
    "OnlineBackup": "Offer a bundled service discount including online backup.",
    "DeviceProtection": "Offer a device-protection bundle promotion.",
    "Dependents": "Consider household/plan-fit review for single-occupant accounts.",
    "Partner": "Consider household/plan-fit review.",
    "StreamingTV": "Review entertainment bundle fit.",
    "StreamingMovies": "Review entertainment bundle fit.",
    "MultipleLines": "Review multi-line plan fit and pricing.",
    "SeniorCitizen": "Consider a senior-friendly support / pricing track.",
    "tenure_group": "Enroll in an early-tenure retention program if in the 0-12 month bucket.",
    "num_services": "Offer a service bundle to increase account stickiness.",
    "has_internet_addons": "Offer an add-on service bundle to increase account stickiness.",
    "gender": "No actionable lever — demographic only.",
    "PhoneService": "Review phone-service bundle fit.",
}


def suggested_action(raw_feature_name):
    base = raw_feature_name.split("__", 1)[-1].rsplit("_", 1)[0]
    return ACTION_MAP.get(base, "Review this factor as part of a manual account check.")


# -------------------------------------------------------------------------
# SIDEBAR — CUSTOMER INPUTS
# -------------------------------------------------------------------------
st.sidebar.markdown("## Customer profile")

st.sidebar.markdown("#### Demographics")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has dependents", ["No", "Yes"])

st.sidebar.markdown("#### Account")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless billing", ["No", "Yes"])
payment = st.sidebar.selectbox(
    "Payment method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
monthly_charges = st.sidebar.slider("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
total_charges = st.sidebar.number_input(
    "Total charges to date ($)", min_value=0.0, value=float(round(monthly_charges * max(tenure, 1), 2))
)

st.sidebar.markdown("#### Services")
phone_service = st.sidebar.selectbox("Phone service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple lines", ["No", "Yes", "No phone service"] if phone_service == "No" else ["No", "Yes"]
)
internet_service = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

if internet_service == "No":
    addon_options = ["No internet service"]
else:
    addon_options = ["No", "Yes"]

online_security = st.sidebar.selectbox("Online security", addon_options)
online_backup = st.sidebar.selectbox("Online backup", addon_options)
device_protection = st.sidebar.selectbox("Device protection", addon_options)
tech_support = st.sidebar.selectbox("Tech support", addon_options)
streaming_tv = st.sidebar.selectbox("Streaming TV", addon_options)
streaming_movies = st.sidebar.selectbox("Streaming movies", addon_options)

predict_clicked = st.sidebar.button("Predict churn risk", type="primary", width='stretch')

# -------------------------------------------------------------------------
# BUILD A SINGLE-ROW DATAFRAME MATCHING THE TRAINING SCHEMA
# -------------------------------------------------------------------------
def build_input_row():
    row = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    df = pd.DataFrame([row])

    # --- replicate train_pipeline.py feature engineering exactly ---
    def tenure_group(t):
        if t <= 12:
            return "0-12"
        if t <= 24:
            return "13-24"
        if t <= 48:
            return "25-48"
        return "49-72"

    df["tenure_group"] = df["tenure"].apply(tenure_group)
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )
    df["has_internet_addons"] = (df["num_services"] > 0).astype(int)
    return df


# -------------------------------------------------------------------------
# HEADER + TABS
# -------------------------------------------------------------------------
st.title("Explainable AI Customer Churn Predictor")
st.caption("Predict churn risk. Understand why. Take informed action.")

tab_predict, tab_global, tab_performance, tab_about = st.tabs(
    ["Prediction", "How the model decides", "Model performance", "About"]
)

# =========================================================================
# TAB 1 — PREDICTION
# =========================================================================
with tab_predict:
    if not predict_clicked:
        st.info("Set a customer profile in the sidebar, then click **Predict churn risk**.")
    else:
        input_df = build_input_row()
        input_p = preprocessor.transform(input_df)

        prob = float(model.predict_proba(input_p)[0, 1])
        pred = int(prob >= THRESHOLD)
        tier = risk_category(prob)
        color = RISK_COLORS[tier]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div class="metric-box"><div class="label">ESTIMATED CHURN PROBABILITY</div>
                <div class="value" style="color:{color}">{prob*100:.1f}%</div></div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="metric-box"><div class="label">RISK CATEGORY</div>
                <div class="value" style="color:{color}">{tier}</div></div>""",
                unsafe_allow_html=True,
            )
        with col3:
            decision_label = "Likely to churn" if pred else "Likely to stay"
            st.markdown(
                f"""<div class="metric-box"><div class="label">MODEL DECISION (threshold {THRESHOLD:.2f})</div>
                <div class="value" style="font-size:20px; color:#E8EDF3; padding-top:6px;">{decision_label}</div></div>""",
                unsafe_allow_html=True,
            )

        st.caption(
            "This is a model estimate based on patterns in historical data, not a certainty about any individual customer."
        )
        st.markdown("---")

        # ---- SHAP explanation for this individual customer ----
        st.subheader("Why is this customer at risk?")
        shap_row = explainer.shap_values(input_p)
        if isinstance(shap_row, list):
            shap_row = shap_row[1]
        shap_row = np.array(shap_row).flatten()

        contrib_df = pd.DataFrame({
            "raw_feature": feature_names,
            "feature": [clean_feature_label(f) for f in feature_names],
            "shap": shap_row,
        })
        contrib_df["abs_shap"] = contrib_df["shap"].abs()
        top_contrib = contrib_df.sort_values("abs_shap", ascending=False).head(10)
        top_contrib = top_contrib.sort_values("shap")

        fig = go.Figure()
        colors = ["#34D399" if v < 0 else "#FF6B4A" for v in top_contrib["shap"]]
        fig.add_trace(go.Bar(
            x=top_contrib["shap"], y=top_contrib["feature"], orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            plot_bgcolor="#0B1420", paper_bgcolor="#0B1420",
            font=dict(color="#E8EDF3"),
            xaxis=dict(title="SHAP value (impact on churn probability)", gridcolor="#223349", zerolinecolor="#8FA1B3"),
            yaxis=dict(gridcolor="#223349"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
        st.plotly_chart(fig, width='stretch')

        col_up, col_down = st.columns(2)
        increasing = contrib_df[contrib_df["shap"] > 0].sort_values("shap", ascending=False).head(5)
        decreasing = contrib_df[contrib_df["shap"] < 0].sort_values("shap").head(5)

        with col_up:
            st.markdown("**Factors increasing churn risk**")
            for _, r in increasing.iterrows():
                st.markdown(
                    f"""<div class="factor-card">↑ <b>{r['feature']}</b><br>
                    <span style="font-size:12.5px;color:#8FA1B3;">SHAP contribution: +{r['shap']:.3f}</span></div>""",
                    unsafe_allow_html=True,
                )
        with col_down:
            st.markdown("**Factors reducing churn risk**")
            if len(decreasing) == 0:
                st.caption("No meaningfully protective factors identified for this profile.")
            for _, r in decreasing.iterrows():
                st.markdown(
                    f"""<div class="factor-card">↓ <b>{r['feature']}</b><br>
                    <span style="font-size:12.5px;color:#8FA1B3;">SHAP contribution: {r['shap']:.3f}</span></div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.subheader("Suggested retention actions")
        st.caption("Generated from the top risk-increasing factors above. Business judgment should confirm these before use.")
        for _, r in increasing.head(3).iterrows():
            action = suggested_action(r["raw_feature"])
            st.markdown(
                f"""<div class="factor-card"><b style="color:#5B9DFF;">{r['feature']}</b><br>
                <span style="font-size:13.5px;">{action}</span></div>""",
                unsafe_allow_html=True,
            )

# =========================================================================
# TAB 2 — GLOBAL MODEL EXPLANATION
# =========================================================================
with tab_global:
    st.subheader("How does the model make predictions?")
    st.caption(
        "Global SHAP feature importance, computed once on the held-out test set. "
        "This shows which features the model relies on most overall — it does not establish causation."
    )

    global_imp = pd.DataFrame(metadata["global_shap_importance"]).head(10)
    global_imp["feature"] = global_imp["feature"].apply(clean_feature_label)

    fig2 = go.Figure(go.Bar(
        x=global_imp["mean_abs_shap"][::-1], y=global_imp["feature"][::-1],
        orientation="h", marker_color="#5B9DFF",
    ))
    fig2.update_layout(
        plot_bgcolor="#0B1420", paper_bgcolor="#0B1420",
        font=dict(color="#E8EDF3"),
        xaxis=dict(title="Mean |SHAP value|", gridcolor="#223349"),
        yaxis=dict(gridcolor="#223349"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Top factors influencing model predictions")
    feature_explainers = {
        "Contract": "The billing contract term. Month-to-month customers churn far more than annual/two-year customers.",
        "tenure": "How long the customer has been with the company. Newer customers churn more.",
        "OnlineSecurity": "Whether the customer has the online security add-on.",
        "TechSupport": "Whether the customer has the tech support add-on.",
        "PaymentMethod": "How the customer pays their bill; electronic check is associated with higher churn.",
        "InternetService": "Type of internet service (fiber optic, DSL, or none).",
        "avg_monthly_spend": "A smoothed spend rate (total charges / tenure).",
        "MonthlyCharges": "The customer's current monthly bill.",
        "PaperlessBilling": "Whether the customer uses paperless billing.",
        "TotalCharges": "Cumulative amount billed to date.",
    }
    for i, row in global_imp.iterrows():
        base_name = row["feature"].split(":")[0].strip()
        desc = feature_explainers.get(base_name, "A feature used by the model to estimate churn risk.")
        st.markdown(f"**{i+1}. {row['feature']}** — {desc}")

# =========================================================================
# TAB 3 — MODEL PERFORMANCE
# =========================================================================
with tab_performance:
    st.subheader("Model performance")
    st.caption(f"Held-out test set — {metadata['n_test']} customers never seen during training or tuning.")

    m = metadata["final_test_metrics"]
    cols = st.columns(4)
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    for c, lab in zip(cols, labels):
        c.markdown(
            f"""<div class="metric-box"><div class="label">{lab.upper()}</div>
            <div class="value">{m[lab]:.3f}</div></div>""", unsafe_allow_html=True,
        )
    cols2 = st.columns(3)
    for c, lab in zip(cols2, ["ROC-AUC", "PR-AUC", "Brier"]):
        c.markdown(
            f"""<div class="metric-box"><div class="label">{lab.upper()}</div>
            <div class="value">{m[lab]:.3f}</div></div>""", unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Model comparison (5-fold cross-validation, training data)")
    comp_df = pd.DataFrame(metadata["model_comparison"])
    st.dataframe(comp_df.style.format({c: "{:.4f}" for c in comp_df.columns if c != "Model"}), width='stretch')

    st.markdown("---")
    st.markdown("#### Selected model & methodology")
    st.markdown(
        f"""
- **Final model:** {metadata['best_model_name']}
- **Hyperparameter search:** Optuna (30 trials), optimizing 5-fold CV ROC-AUC
- **Best CV ROC-AUC found:** {metadata['optuna_best_cv_roc_auc']:.4f}
- **Calibration:** Isotonic regression, 5-fold — train Brier score {metadata['brier_score_train']:.4f}
- **Decision threshold:** {THRESHOLD:.2f}, chosen by maximizing F1 on the (non-resampled) training split — not the default 0.50
- **Class imbalance handling:** SMOTE, applied only to the training split, after the train/test split
- **Selection criterion:** models were compared on ROC-AUC, PR-AUC, precision, recall, and F1. For churn prediction, missing an actual churner (false negative) is usually costlier than a false alarm, so recall and ROC-AUC/PR-AUC were weighted more heavily than raw accuracy when interpreting the comparison above.
        """
    )

# =========================================================================
# TAB 4 — ABOUT
# =========================================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
This application predicts telecom customer churn and explains each prediction
using SHAP (SHapley Additive exPlanations), so the output is useful to a
non-technical retention team, not just a data scientist.

**Dataset:** IBM Telco Customer Churn (~7,043 customers, publicly available).

**Pipeline:** data cleaning → feature engineering → train/test split →
preprocessing → SMOTE (train-only) → model comparison → Optuna hyperparameter
tuning → isotonic probability calibration → threshold optimization → SHAP
explainability → saved model artifacts. The pipeline is run once, offline
(`train_pipeline.py`); this app only ever loads the saved artifacts.

**Stack:** Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Optuna,
imbalanced-learn, Streamlit, Plotly.
        """
    )
    st.markdown(
        """<div class="disclaimer">
        This tool produces a statistical estimate based on historical patterns
        in a public benchmark dataset. It is a portfolio / demonstration
        project and is not intended for use in real customer retention
        decisions without further validation on your own data.
        </div>""",
        unsafe_allow_html=True,
    )
