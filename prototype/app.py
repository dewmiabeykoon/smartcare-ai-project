"""
SmartCare Hospital - Appointment No-Show Prediction Prototype
Task 08 - AI Prototype Development (CCS3440)

Run with:  streamlit run app.py
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===========================================================
# PAGE CONFIG + GLOBAL STYLE
# ===========================================================
st.set_page_config(
    page_title="SmartCare | No-Show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #f7f9fb; }

    .main-header {
        background: linear-gradient(90deg, #0f4c81 0%, #1a7a9e 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.4rem;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    section[data-testid="stSidebar"] {
        background-color: #0f2540;
    }
    section[data-testid="stSidebar"] * { color: #eaf1f8 !important; }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e8ee;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    /* fix: metric cards have a white background even inside the dark sidebar,
       so their text needs to stay dark, overriding the sidebar-wide light text rule above */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] * {
        color: #0f2540 !important;
    }

    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }
    .result-noshow {
        background-color: #fdecec;
        border-left: 6px solid #d64545;
    }
    .result-attend {
        background-color: #eaf7ee;
        border-left: 6px solid #2e9e5b;
    }
    .result-card h3 { margin-top: 0; }

    .reason-chip {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.82rem;
        margin: 0.15rem 0.3rem 0.15rem 0;
        font-weight: 500;
    }
    .chip-up { background-color: #fde3e3; color: #a02323; }
    .chip-down { background-color: #dff2e6; color: #1f7a44; }

    div[data-testid="stForm"] {
        background-color: white;
        border: 1px solid #e3e8ee;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
    }

    /* force readable dark text in the main content area regardless of the
       user's OS/browser dark-mode setting, which Streamlit otherwise inherits */
    div[data-testid="stAppViewContainer"] > div.main label,
    div[data-testid="stAppViewContainer"] > div.main p,
    div[data-testid="stAppViewContainer"] > div.main span,
    div[data-testid="stAppViewContainer"] > div.main h3,
    div[data-testid="stAppViewContainer"] > div.main h4,
    div[data-testid="stAppViewContainer"] > div.main .stMarkdown {
        color: #0f2540 !important;
    }
    div[data-testid="stAppViewContainer"] > div.main input,
    div[data-testid="stAppViewContainer"] > div.main select,
    div[data-testid="stAppViewContainer"] > div.main textarea {
        color: #0f2540 !important;
        background-color: #ffffff !important;
    }

    h3 { color: #0f2540; }
    hr { margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ===========================================================
# LOAD PIPELINE (model + encoders + scaler + SHAP explainer)
# ===========================================================
@st.cache_resource
def load_pipeline():
    raw = pd.read_csv("smartcare_ai_dataset_1000.csv")
    df = raw.copy()

    # --- missing value handling (same as corrected Task 03) ---
    df.loc[df["admitted"] == 0, "room_type"] = df.loc[
        df["admitted"] == 0, "room_type"
    ].fillna("Not Applicable")
    admitted_mode = df.loc[df["admitted"] == 1, "room_type"].mode()[0]
    df.loc[df["admitted"] == 1, "room_type"] = df.loc[
        df["admitted"] == 1, "room_type"
    ].fillna(admitted_mode)

    num_cols_raw = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols_raw:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols_raw = df.select_dtypes(include=["object"]).columns
    for col in cat_cols_raw:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    # --- outlier clipping (target/id excluded) ---
    exclude_from_outlier = ["record_id", "no_show", "readmitted_30_days"]
    outlier_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(
        exclude_from_outlier
    )
    for col in outlier_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

    # --- feature engineering (same as Task 03) ---
    df["Age_Group"] = pd.cut(
        df["age"], bins=[0, 18, 35, 50, 65, 100],
        labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"],
    )
    df["High_BP"] = ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)
    df["Missed_Rate"] = df["missed_previous_appointments"] / (df["previous_appointments"] + 1)

    feature_cols = [
        "age", "gender", "blood_group", "department", "diagnosis",
        "waiting_days", "previous_appointments", "missed_previous_appointments",
        "admitted", "room_type", "length_of_stay_days", "previous_admissions",
        "systolic_bp", "diastolic_bp", "blood_sugar_mg_dl", "cholesterol_mg_dl",
        "bmi", "lab_tests_count", "treatments_count", "consultation_fee_lkr",
        "room_charge_lkr", "lab_charge_lkr", "medicine_charge_lkr", "total_bill_lkr",
        "payment_status", "payment_method", "Age_Group", "High_BP", "Missed_Rate",
    ]
    df = df[feature_cols].copy()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(df[feature_cols])

    model = joblib.load("logistic_regression.joblib")

    # SHAP explainer, using real training data as background (same feature space)
    split = joblib.load("train_test_split.joblib")
    X_train = split["X_train"][feature_cols]
    background = shap.sample(X_train, 100, random_state=42)

    # shap.Explainer can't call a scikit-learn Pipeline object directly — it
    # needs a plain function. Wrap predict_proba so SHAP explains the
    # predicted probability of class 1 (No-Show).
    def model_predict_noshow_proba(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.Explainer(model_predict_noshow_proba, background, feature_names=feature_cols)

    # precomputed global SHAP / coefficient importance (Task 07 output)
    try:
        global_importance = pd.read_csv("shap_feature_importance.csv")
    except FileNotFoundError:
        global_importance = None

    try:
        eval_results = pd.read_csv("model_evaluation_results.csv")
    except FileNotFoundError:
        eval_results = None

    raw_display = raw.copy()
    raw_display.loc[raw_display["admitted"] == 0, "room_type"] = raw_display.loc[
        raw_display["admitted"] == 0, "room_type"
    ].fillna("Not Applicable")
    raw_display.loc[raw_display["admitted"] == 1, "room_type"] = raw_display.loc[
        raw_display["admitted"] == 1, "room_type"
    ].fillna(admitted_mode)

    return model, encoders, scaler, feature_cols, raw_display, explainer, global_importance, eval_results


model, encoders, scaler, feature_cols, raw_df, explainer, global_importance, eval_results = load_pipeline()

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ===========================================================
# SIDEBAR
# ===========================================================
with st.sidebar:
    st.markdown("## 🏥 SmartCare Hospital")
    st.caption(" AI Coursework Prototype")
    st.markdown("---")
    st.markdown("### Model in use")
    st.markdown("**Logistic Regression**")
    st.caption("Selected as best model using highest Accuracy, Precision, Recall and F1")

    if eval_results is not None:
        best_row = eval_results.sort_values("F1 Score", ascending=False).iloc[0]
        st.metric("Accuracy", f"{best_row['Accuracy']*100:.1f}%")
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{best_row['F1 Score']*100:.1f}%")
        c2.metric("ROC-AUC", f"{best_row['ROC-AUC']*100:.1f}%")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "Predicts whether a patient is likely to miss their hospital appointment, "
        "using 29 patient, clinical, operational and financial features from the "
        "SmartCare Hospital AI Dataset."
    )
    st.caption("⚠️ Decision-support tool only — not a substitute for clinical judgement.")


# ===========================================================
# HEADER
# ===========================================================
st.markdown("""
<div class="main-header">
    <h1>🏥 SmartCare Appointment No-Show Predictor</h1>
    <p>AI-powered decision support · Enter patient details to predict attendance likelihood</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_explain, tab_about = st.tabs([" Predict", "  Explainability (SHAP)", "ℹ️  About the Project"])


# ===========================================================
# TAB 1 — PREDICT
# ===========================================================
with tab_predict:
    with st.form("patient_form"):
        st.markdown("#### 👤 Patient Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=100, value=40)
        with c2:
            gender = st.selectbox("Gender", sorted(raw_df["gender"].unique()))
        with c3:
            blood_group = st.selectbox("Blood Group", sorted(raw_df["blood_group"].unique()))

        st.markdown("#### 📅 Appointment Details")
        c1, c2 = st.columns(2)
        with c1:
            department = st.selectbox("Department", sorted(raw_df["department"].unique()))
            waiting_days = st.number_input("Waiting Days (until appointment)", min_value=0, max_value=90, value=15)
            previous_appointments = st.number_input("Previous Appointments", min_value=0, max_value=50, value=5)
        with c2:
            diagnosis = st.selectbox("Diagnosis", sorted(raw_df["diagnosis"].unique()))
            missed_previous_appointments = st.number_input("Missed Previous Appointments", min_value=0, max_value=50, value=1)
            previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=0)

        st.markdown("#### 🩺 Clinical Measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=220, value=120)
            blood_sugar = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=100)
        with c2:
            diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=140, value=80)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        with c3:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)

        st.markdown("#### 🛏️ Hospital Stay & Admission")
        c1, c2, c3 = st.columns(3)
        with c1:
            admitted = st.selectbox("Admitted?", ["No", "Yes"])
        with c2:
            if admitted == "No":
                room_type = "Not Applicable"
                st.text_input("Room Type", value="Not Applicable", disabled=True)
            else:
                real_room_types = sorted(t for t in raw_df["room_type"].unique() if t != "Not Applicable")
                room_type = st.selectbox("Room Type", real_room_types)
        with c3:
            length_of_stay_days = st.number_input("Length of Stay (days)", min_value=0, max_value=60, value=0)

        st.markdown("#### 🧪 Labs & Treatment")
        c1, c2 = st.columns(2)
        with c1:
            lab_tests_count = st.number_input("Lab Tests Count", min_value=0, max_value=30, value=2)
        with c2:
            treatments_count = st.number_input("Treatments Count", min_value=0, max_value=30, value=1)

        st.markdown("#### 💳 Billing")
        c1, c2, c3 = st.columns(3)
        with c1:
            consultation_fee = st.number_input("Consultation Fee (LKR)", min_value=0, value=2000)
            room_charge = st.number_input("Room Charge (LKR)", min_value=0, value=0)
        with c2:
            lab_charge = st.number_input("Lab Charge (LKR)", min_value=0, value=3000)
            medicine_charge = st.number_input("Medicine Charge (LKR)", min_value=0, value=4000)
        with c3:
            payment_status = st.selectbox("Payment Status", sorted(raw_df["payment_status"].unique()))
            payment_method = st.selectbox("Payment Method", sorted(raw_df["payment_method"].unique()))

        total_bill = consultation_fee + room_charge + lab_charge + medicine_charge
        st.caption(f"Total Bill (auto-calculated): **LKR {total_bill:,.0f}**")

        submitted = st.form_submit_button("🔍 Predict Attendance", use_container_width=True)

    if submitted:
        if age <= 18: age_group = "Child"
        elif age <= 35: age_group = "Young Adult"
        elif age <= 50: age_group = "Adult"
        elif age <= 65: age_group = "Middle Age"
        else: age_group = "Senior"

        high_bp = int(systolic_bp >= 140 or diastolic_bp >= 90)
        missed_rate = missed_previous_appointments / (previous_appointments + 1)

        raw_input = {
            "age": age, "gender": gender, "blood_group": blood_group,
            "department": department, "diagnosis": diagnosis, "waiting_days": waiting_days,
            "previous_appointments": previous_appointments,
            "missed_previous_appointments": missed_previous_appointments,
            "admitted": 1 if admitted == "Yes" else 0, "room_type": room_type,
            "length_of_stay_days": length_of_stay_days, "previous_admissions": previous_admissions,
            "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
            "blood_sugar_mg_dl": blood_sugar, "cholesterol_mg_dl": cholesterol, "bmi": bmi,
            "lab_tests_count": lab_tests_count, "treatments_count": treatments_count,
            "consultation_fee_lkr": consultation_fee, "room_charge_lkr": room_charge,
            "lab_charge_lkr": lab_charge, "medicine_charge_lkr": medicine_charge,
            "total_bill_lkr": total_bill, "payment_status": payment_status,
            "payment_method": payment_method, "Age_Group": age_group,
            "High_BP": high_bp, "Missed_Rate": missed_rate,
        }

        input_df = pd.DataFrame([raw_input])[feature_cols]

        for col, le in encoders.items():
            val = str(input_df.at[0, col])
            if val not in le.classes_:
                val = le.classes_[0]
            input_df[col] = le.transform([val])

        input_scaled = pd.DataFrame(scaler.transform(input_df[feature_cols]), columns=feature_cols)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        # live local SHAP explanation for this patient
        shap_exp = explainer(input_scaled)
        local_shap = pd.Series(shap_exp.values[0], index=feature_cols).sort_values(key=abs, ascending=False)

        st.session_state.last_result = {
            "pred": pred, "proba": proba, "input_df": input_df,
            "shap_exp": shap_exp, "local_shap": local_shap,
        }

    if st.session_state.last_result:
        r = st.session_state.last_result
        css_class = "result-noshow" if r["pred"] == 1 else "result-attend"
        label = "⚠️ Likely NO-SHOW" if r["pred"] == 1 else "✅ Likely to ATTEND"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <h3>{label}</h3>
            <p>Probability of missing this appointment: <b>{r['proba']*100:.1f}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(r["proba"], 0.0), 1.0), text="No-show risk")

        st.markdown("**Top factors behind this prediction:**")
        top5 = r["local_shap"].head(5)
        chips_html = ""
        for feat, val in top5.items():
            cls = "chip-up" if val > 0 else "chip-down"
            arrow = "↑ raises risk" if val > 0 else "↓ lowers risk"
            chips_html += f'<span class="reason-chip {cls}">{feat} {arrow}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)
        st.caption("See the **Explainability (SHAP)** tab for the full breakdown of this prediction.")

        with st.expander("View processed model input"):
            st.dataframe(r["input_df"].T.rename(columns={0: "value"}))


# ===========================================================
# TAB 2 — EXPLAINABILITY
# ===========================================================
with tab_explain:
    st.markdown("#### 🌍 Global Feature Importance")
    st.caption(
        "Which features matter most to the model overall, computed with SHAP on the "
        "training set. Positive = pushes predictions toward **No-Show**, "
        "negative = pushes toward **Attended**."
    )

    if global_importance is not None:
        gi = global_importance.copy()
        val_col = "Coefficient" if "Coefficient" in gi.columns else gi.columns[1]
        gi = gi.reindex(gi[val_col].abs().sort_values(ascending=False).index).head(10)
        gi = gi.iloc[::-1]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = ["#d64545" if v > 0 else "#2e9e5b" for v in gi[val_col]]
        ax.barh(gi["Feature"], gi[val_col], color=colors)
        ax.set_xlabel("SHAP value (impact on No-Show prediction)")
        ax.axvline(0, color="#333", linewidth=0.8)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("`shap_feature_importance.csv` not found next to app.py — global chart unavailable.")

    st.markdown("---")
    st.markdown("#### 🔎 Patient-Specific Explanation")

    if st.session_state.last_result is None:
        st.info("Make a prediction in the **Predict** tab to see a patient-specific SHAP breakdown here.")
    else:
        r = st.session_state.last_result
        st.caption(
            f"Why the model predicted **{'No-Show' if r['pred']==1 else 'Attended'}** "
            f"for this patient ({r['proba']*100:.1f}% no-show probability):"
        )
        fig2 = plt.figure()
        shap.plots.waterfall(r["shap_exp"][0], max_display=12, show=False)
        fig2 = plt.gcf()
        fig2.set_size_inches(10, 6)
        st.pyplot(fig2, bbox_inches="tight")
        plt.close(fig2)

        st.caption(
            "Bars pushing right (red) increase the predicted probability of a no-show; "
            "bars pushing left (blue) decrease it. `E[f(x)]` is the model's average "
            "prediction across all patients; `f(x)` is the prediction for this patient."
        )


# ===========================================================
# TAB 3 — ABOUT
# ===========================================================
with tab_about:
    st.markdown("""
#### About this prototype
This is the AI prototype for the **CCS3440 Artificial Intelligence** coursework,
built on the **SmartCare Hospital AI Dataset** (Option A — Appointment No-Show Prediction).

**Pipeline:**
1. Missing value handling, duplicate removal, and IQR-based outlier clipping
2. Feature engineering — `Age_Group`, `High_BP`, `Missed_Rate`
3. Label encoding + standard scaling 
4. Five models trained and tuned with `GridSearchCV`: Logistic Regression, Decision Tree,
   Random Forest, KNN, XGBoost
5. **Logistic Regression** selected as the best model (highest Accuracy / Precision / Recall / F1)
6. SHAP used for both global and per-patient explainability

**Limitations:**
- Model performance is moderate (~63% accuracy); this tool is meant to **support**, not
  replace, staff judgement when deciding who to prioritise for reminder calls.
- Trained on a synthetic 1,000-record dataset — not validated on real hospital data.
    """)

    if eval_results is not None:
        st.markdown("#### Model Comparison (Task 06)")
        st.dataframe(
            eval_results.style.format({
                "Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}",
                "F1 Score": "{:.1%}", "ROC-AUC": "{:.1%}",
            }),
            use_container_width=True,
        )

st.markdown("---")
st.caption(" SmartCare Hospital Prediction System | 2026")
