"""
SmartCare AI - Appointment No-Show Prediction
A healthcare AI dashboard prototype (Streamlit) for the AI coursework
(Option A: Appointment No-Show Prediction).

Flow: User Input -> Process Patient Information -> Generate Prediction -> Display Result
"""

import warnings

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.utils import (
    load_custom_css, DEFAULT_RISK_THRESHOLDS, get_risk_level, RISK_COLORS,
    RECOMMENDATIONS, DEPARTMENT_NAMES, DIAGNOSIS_NAMES, GENDER_MAP,
    BLOOD_GROUP_MAP, DEPARTMENT_MAP, DIAGNOSIS_MAP, FEATURE_LABELS,
    CLEAN_DATASET_PATH, EVAL_RESULTS_PATH, PRIMARY, ACCENT, DANGER, SUCCESS,
    WARNING, PLOTLY_TEMPLATE,
)
from src.preprocessing import build_feature_row, InputValidationError
from src.prediction import predict, get_model_status, PredictionError
from src.explainability import (
    compute_shap_for_instance, load_global_shap_importance,
    generate_explanation_text, ExplainabilityError,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="SmartCare AI | No-Show Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(load_custom_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "risk_thresholds" not in st.session_state:
    st.session_state.risk_thresholds = dict(DEFAULT_RISK_THRESHOLDS)
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "feature_row" not in st.session_state:
    st.session_state.feature_row = None
if "raw_input" not in st.session_state:
    st.session_state.raw_input = None
if "shap_df" not in st.session_state:
    st.session_state.shap_df = None
if "shap_error" not in st.session_state:
    st.session_state.shap_error = None


@st.cache_data(show_spinner=False)
def load_clean_dataset():
    return pd.read_csv(CLEAN_DATASET_PATH)


@st.cache_data(show_spinner=False)
def load_eval_results():
    return pd.read_csv(EVAL_RESULTS_PATH)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding: .6rem 0 1rem 0;'>"
            "<div style='font-size:2.4rem;'>🏥</div>"
            "<div style='font-size:1.15rem; font-weight:800;'>SmartCare AI</div>"
            "<div style='font-size:.78rem; opacity:.85;'>No-Show Prediction Suite</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = {
            "Dashboard": "📊 Dashboard",
            "Patient Prediction": "🧾 Patient Prediction",
            "Prediction Result": "🎯 Prediction Result",
            "Explainability": "🧠 Explainability (XAI)",
            "About Model": "ℹ️ About Model",
        }
        choice = st.radio(
            "Navigate",
            list(pages.keys()),
            format_func=lambda k: pages[k],
            index=list(pages.keys()).index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = choice

        st.markdown("---")
        with st.expander("⚙️ Risk Threshold Settings"):
            st.caption("Configurable decision boundaries for risk labeling.")
            low = st.slider("Low / Medium boundary", 0.05, 0.60,
                             st.session_state.risk_thresholds["low"], 0.05)
            high = st.slider("Medium / High boundary", 0.40, 0.95,
                              st.session_state.risk_thresholds["high"], 0.05)
            if high <= low:
                st.warning("High boundary must exceed low boundary.")
            else:
                st.session_state.risk_thresholds = {"low": low, "high": high}

        status = get_model_status()
        st.markdown("---")
        if status["ok"]:
            st.markdown("🟢 **Model Status:** Online")
        else:
            st.markdown("🔴 **Model Status:** Error")


NAV_PAGES = {
    "Dashboard": "📊  Dashboard",
    "Patient Prediction": "🧾  Prediction",
    "Prediction Result": "🎯  Result",
    "Explainability": "🧠  XAI",
    "About Model": "ℹ️  About",
}


def render_topnav():
    """Pill-style navigation bar shown right under the hero on every page,
    so users can move between Dashboard / Prediction / Result / XAI / About
    without needing the sidebar."""
    with st.container(key="topnav_bar"):
        cols = st.columns(len(NAV_PAGES))
        for col, (key, label) in zip(cols, NAV_PAGES.items()):
            is_active = st.session_state.page == key
            with col:
                if st.button(
                    label, key=f"topnav_{key}",
                    type="primary" if is_active else "secondary",
                    width='stretch',
                ):
                    st.session_state.page = key
                    st.rerun()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def page_dashboard():
    st.markdown(
        """
        <div class="hero">
            <h1>SmartCare AI — Appointment No-Show Prediction</h1>
            <p>An AI-powered clinical decision-support prototype that predicts whether a patient
            is likely to attend or miss their upcoming appointment, so care teams can act early
            and reduce avoidable no-shows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_topnav()

    status = get_model_status()
    df = load_clean_dataset()
    eval_df = load_eval_results()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Model Status</div>
            <div class="stat-value" style="font-size:1.2rem;">
            {"🟢 Online" if status["ok"] else "🔴 Offline"}</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        no_show_rate = df["no_show"].mean() * 100
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Historical No-Show Rate</div>
            <div class="stat-value">{no_show_rate:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        best_row = eval_df.sort_values("F1 Score", ascending=False).iloc[0]
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Best Model (F1)</div>
            <div class="stat-value" style="font-size:1.2rem;">{best_row['Model']}</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="stat-card"><div class="stat-label">Patients in Dataset</div>
            <div class="stat-value">{len(df):,}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Prediction Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.3])

    with c1:
        counts = df["no_show"].value_counts().rename({0: "Attended", 1: "No-Show"})
        fig = go.Figure(
            data=[go.Pie(
                labels=counts.index, values=counts.values, hole=0.58,
                marker=dict(colors=[SUCCESS, DANGER]),
                textinfo="label+percent", textfont=dict(size=13),
            )]
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False, height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Appointment Status Distribution",
        )
        st.plotly_chart(fig, width='stretch')

    with c2:
        dept_rate = (
            df.assign(department_name=df["department"].map(DEPARTMENT_NAMES))
            .groupby("department_name")["no_show"].mean().sort_values(ascending=True) * 100
        )
        fig2 = go.Figure(go.Bar(
            x=dept_rate.values, y=dept_rate.index, orientation="h",
            marker=dict(color=dept_rate.values, colorscale=[[0, PRIMARY], [1, DANGER]]),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            title="Department-wise No-Show Rate (%)",
            xaxis_title="No-Show Rate (%)", yaxis_title="",
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown('<div class="section-title">Quick Access</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(
            '<div class="card"><b>🧾 New Prediction</b><br>'
            'Enter patient details and get an instant no-show risk prediction.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Go to Patient Prediction →", width='stretch'):
            st.session_state.page = "Patient Prediction"
            st.rerun()
    with q2:
        st.markdown(
            '<div class="card"><b>🧠 Explainability</b><br>'
            'See exactly why the model made its most recent prediction.</div>',
            unsafe_allow_html=True,
        )
        if st.button("View Explainability →", width='stretch'):
            st.session_state.page = "Explainability"
            st.rerun()
    with q3:
        st.markdown(
            '<div class="card"><b>ℹ️ About the Model</b><br>'
            'Review evaluation metrics and training details.</div>',
            unsafe_allow_html=True,
        )
        if st.button("View Model Info →", width='stretch'):
            st.session_state.page = "About Model"
            st.rerun()


# ---------------------------------------------------------------------------
# Patient Prediction form
# ---------------------------------------------------------------------------
def page_prediction_form():
    st.markdown(
        """
        <div class="hero">
            <h1>Patient Prediction</h1>
            <p>Enter the patient's information below. Only fields actually used by the
            trained model are requested — nothing else.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_topnav()

    with st.form("patient_form", clear_on_submit=False):
        st.markdown('<div class="form-group-title">👤 Patient Information</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        age = p1.number_input("Age", min_value=0, max_value=120, value=45)
        gender = p2.selectbox("Gender", list(GENDER_MAP.keys()))
        blood_group = p3.selectbox("Blood Group", list(BLOOD_GROUP_MAP.keys()))

        st.markdown('<div class="form-group-title">🩺 Clinical Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        diagnosis = c1.selectbox("Diagnosis", list(DIAGNOSIS_MAP.keys()))
        systolic_bp = c2.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=128)
        diastolic_bp = c3.number_input("Diastolic BP (mmHg)", min_value=30, max_value=180, value=82)
        c4, c5, c6 = st.columns(3)
        blood_sugar = c4.number_input("Blood Sugar (mg/dL)", min_value=40, max_value=600, value=110)
        cholesterol = c5.number_input("Cholesterol (mg/dL)", min_value=80, max_value=500, value=190)
        bmi = c6.number_input("BMI", min_value=8.0, max_value=70.0, value=24.5, step=0.1)

        st.markdown('<div class="form-group-title">📅 Appointment / Hospital Information</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        department = a1.selectbox("Department", list(DEPARTMENT_MAP.keys()))
        waiting_days = a2.number_input("Waiting Days (booking → appointment)", min_value=0, max_value=365, value=7)
        previous_admissions = a3.number_input("Previous Admissions", min_value=0, max_value=100, value=0)
        a4, a5 = st.columns(2)
        previous_appointments = a4.number_input("Previous Appointments", min_value=0, max_value=200, value=3)
        missed_previous = a5.number_input(
            "Previously Missed Appointments", min_value=0, max_value=200, value=0,
            help="Must not exceed 'Previous Appointments'.",
        )

        st.markdown('<div class="form-group-title">💳 Financial Information</div>', unsafe_allow_html=True)
        f1, _, _ = st.columns(3)
        consultation_fee = f1.number_input("Consultation Fee (LKR)", min_value=0, max_value=100000, value=2000)

        submitted = st.form_submit_button("🔮 Predict Attendance", width='stretch')

    if submitted:
        raw = {
            "age": int(age), "gender": gender, "blood_group": blood_group,
            "department": department, "diagnosis": diagnosis,
            "waiting_days": int(waiting_days),
            "previous_appointments": int(previous_appointments),
            "missed_previous_appointments": int(missed_previous),
            "previous_admissions": int(previous_admissions),
            "systolic_bp": int(systolic_bp), "diastolic_bp": int(diastolic_bp),
            "blood_sugar_mg_dl": float(blood_sugar), "cholesterol_mg_dl": float(cholesterol),
            "bmi": float(bmi), "consultation_fee_lkr": int(consultation_fee),
        }

        with st.spinner("Processing patient information and running the model..."):
            try:
                feature_row = build_feature_row(raw)
                result = predict(feature_row)
            except InputValidationError as exc:
                st.error(f"⚠️ Please check your input: {exc}")
                return
            except PredictionError as exc:
                st.error(f"⚠️ Prediction failed: {exc}")
                return

            st.session_state.raw_input = raw
            st.session_state.feature_row = feature_row
            st.session_state.prediction_result = result

            try:
                st.session_state.shap_df = compute_shap_for_instance(feature_row)
                st.session_state.shap_error = None
            except ExplainabilityError as exc:
                st.session_state.shap_df = None
                st.session_state.shap_error = str(exc)

        st.success("Prediction generated successfully.")
        st.session_state.page = "Prediction Result"
        st.rerun()


# ---------------------------------------------------------------------------
# Prediction Result
# ---------------------------------------------------------------------------
def page_result():
    st.markdown(
        """
        <div class="hero">
            <h1>Prediction Result</h1>
            <p>Outcome of the most recent patient submitted for prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_topnav()

    result = st.session_state.prediction_result
    if result is None:
        st.info("No prediction yet. Go to **Patient Prediction** to submit patient details.")
        if st.button("Go to Patient Prediction →"):
            st.session_state.page = "Patient Prediction"
            st.rerun()
        return

    no_show_p = result["no_show_probability"]
    attended_p = result["attended_probability"]
    thresholds = st.session_state.risk_thresholds
    risk = get_risk_level(no_show_p, thresholds)
    color = RISK_COLORS[risk]

    is_no_show = result["prediction"] == 1
    headline = "⚠️ High Risk of No-Show" if is_no_show else "✅ Appointment Likely Attended"
    grad = f"linear-gradient(120deg, {color} 0%, {'#7f1d1d' if is_no_show else '#065f46'} 100%)"

    st.markdown(
        f"""
        <div class="result-card" style="background:{grad};">
            <h2>{headline}</h2>
            <p>No-show probability: <b>{no_show_p*100:.1f}%</b> &nbsp;|&nbsp; Attendance probability: <b>{attended_p*100:.1f}%</b></p>
            <div class="risk-pill">{risk}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=no_show_p * 100,
            number={"suffix": "%"},
            title={"text": "No-Show Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, thresholds["low"] * 100], "color": "#dcfce7"},
                    {"range": [thresholds["low"] * 100, thresholds["high"] * 100], "color": "#fef3c7"},
                    {"range": [thresholds["high"] * 100, 100], "color": "#fee2e2"},
                ],
            },
        ))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig, width='stretch')

    with c2:
        fig2 = go.Figure(go.Bar(
            x=["Attended", "No-Show"], y=[attended_p * 100, no_show_p * 100],
            marker=dict(color=[SUCCESS, DANGER]),
            text=[f"{attended_p*100:.1f}%", f"{no_show_p*100:.1f}%"], textposition="outside",
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=300, margin=dict(l=20, r=20, t=50, b=10),
            title="Class Probability Comparison", yaxis_title="Probability (%)", yaxis_range=[0, 105],
        )
        st.plotly_chart(fig2, width='stretch')

    st.markdown('<div class="section-title">Confidence &amp; Explanation</div>', unsafe_allow_html=True)
    confidence = max(attended_p, no_show_p)
    if st.session_state.shap_df is not None:
        explanation = generate_explanation_text(st.session_state.shap_df, no_show_p, result["prediction"])
    else:
        explanation = (
            f"Model confidence for this prediction is {confidence*100:.1f}%. "
            "Detailed feature-level explanation is unavailable "
            f"({st.session_state.shap_error})." if st.session_state.shap_error else
            f"Model confidence for this prediction is {confidence*100:.1f}%."
        )
    st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

    if st.session_state.shap_df is not None:
        st.markdown('<div class="section-title">Top Contributing Features</div>', unsafe_allow_html=True)
        top = st.session_state.shap_df.head(6).sort_values("shap_value")
        colors = [DANGER if v > 0 else PRIMARY for v in top["shap_value"]]
        fig3 = go.Figure(go.Bar(
            x=top["shap_value"], y=top["label"], orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig3.update_layout(
            template=PLOTLY_TEMPLATE, height=340, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="SHAP contribution (→ increases No-Show risk)",
        )
        st.plotly_chart(fig3, width='stretch')

    st.button(
        "🧠 View Full Explainability Report →",
        on_click=lambda: st.session_state.update(page="Explainability"),
        width='stretch',
    )


# ---------------------------------------------------------------------------
# Explainability (XAI)
# ---------------------------------------------------------------------------
def page_explainability():
    st.markdown(
        """
        <div class="hero">
            <h1>Explainable AI</h1>
            <p>Why did the model predict this? All values below are computed live from the
            actual trained model using SHAP - nothing is fabricated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_topnav()

    result = st.session_state.prediction_result
    if result is None:
        st.info("No prediction yet. Go to **Patient Prediction** to submit patient details first.")
        if st.button("Go to Patient Prediction →"):
            st.session_state.page = "Patient Prediction"
            st.rerun()
        return

    no_show_p = result["no_show_probability"]
    thresholds = st.session_state.risk_thresholds
    risk = get_risk_level(no_show_p, thresholds)
    color = RISK_COLORS[risk]

    # 1. Prediction result recap
    st.markdown(
        f"""<div class="card"><b>Prediction:</b> {"High Risk of No-Show" if result['prediction']==1 else "Appointment Attended"}
        &nbsp;|&nbsp; <b>No-Show Probability:</b> {no_show_p*100:.1f}%
        &nbsp;|&nbsp; <span class="risk-pill" style="background:{color}22; color:{color};">{risk}</span></div>""",
        unsafe_allow_html=True,
    )

    # 2. Probability visualization
    st.markdown('<div class="section-title">Probability Visualization</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Prediction"], x=[result["attended_probability"] * 100],
        name="Attended", orientation="h", marker_color=SUCCESS,
        text=f"Attended {result['attended_probability']*100:.1f}%", textposition="inside",
    ))
    fig.add_trace(go.Bar(
        y=["Prediction"], x=[result["no_show_probability"] * 100],
        name="No-Show", orientation="h", marker_color=DANGER,
        text=f"No-Show {result['no_show_probability']*100:.1f}%", textposition="inside",
    ))
    fig.update_layout(
        barmode="stack", template=PLOTLY_TEMPLATE, height=180,
        margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
        xaxis_title="Probability (%)",
    )
    st.plotly_chart(fig, width='stretch')

    # 3. SHAP feature contribution
    st.markdown('<div class="section-title">🔍 Why did the AI predict this?</div>', unsafe_allow_html=True)
    if st.session_state.shap_df is None:
        st.warning(
            "A live SHAP explanation could not be computed for this prediction"
            + (f": {st.session_state.shap_error}" if st.session_state.shap_error else ".")
        )
    else:
        shap_df = st.session_state.shap_df
        top = shap_df.head(10).sort_values("shap_value")
        colors = [DANGER if v > 0 else PRIMARY for v in top["shap_value"]]
        fig2 = go.Figure(go.Bar(
            x=top["shap_value"], y=top["label"], orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>SHAP contribution: %{x:.3f}<extra></extra>",
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=420, margin=dict(l=10, r=10, t=10, b=40),
            xaxis_title="SHAP value (log-odds impact on No-Show prediction)",
            annotations=[
                dict(x=0.02, y=-0.16, xref="paper", yref="paper", showarrow=False,
                     text="🔴 Increases No-Show risk", font=dict(color=DANGER, size=12)),
                dict(x=0.98, y=-0.16, xref="paper", yref="paper", showarrow=False,
                     text="🔵 Reduces No-Show risk", font=dict(color=PRIMARY, size=12)),
            ],
        )
        st.plotly_chart(fig2, width='stretch')

        st.markdown('<div class="section-title">Top Contributing Factors</div>', unsafe_allow_html=True)
        display_df = shap_df.head(8)[["label", "shap_value"]].rename(
            columns={"label": "Feature", "shap_value": "SHAP Contribution"}
        )
        st.dataframe(display_df, width='stretch', hide_index=True)

        st.markdown('<div class="section-title">Prediction Explanation Summary</div>', unsafe_allow_html=True)
        explanation = generate_explanation_text(shap_df, no_show_p, result["prediction"])
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

    # 4. Recommendations
    st.markdown('<div class="section-title">✅ Recommended Actions</div>', unsafe_allow_html=True)
    st.caption("Focused on appointment management only — not medical advice. "
               "These recommendations do not guarantee prevention of a no-show.")
    for rec in RECOMMENDATIONS[risk]:
        st.markdown(f'<div class="rec-item">• {rec}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# About Model
# ---------------------------------------------------------------------------
def page_about_model():
    st.markdown(
        """
        <div class="hero">
            <h1>About the Model</h1>
            <p>Technical details and evaluation results for the deployed prediction model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_topnav()

    eval_df = load_eval_results()
    df = load_clean_dataset()
    best_row = eval_df.sort_values("F1 Score", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    info = [
        ("Deployed Model", "Logistic Regression"),
        ("Problem Type", "Binary Classification"),
        ("Target Variable", "no_show"),
        ("Best F1 Score Model", best_row["Model"]),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], info):
        col.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value" style="font-size:1.15rem;">{val}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)
    st.dataframe(eval_df, width='stretch', hide_index=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(name=metric, x=eval_df["Model"], y=eval_df[metric]))
    fig.update_layout(
        barmode="group", template=PLOTLY_TEMPLATE, height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Model Performance Comparison", yaxis_title="Score", legend_title="Metric",
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
    try:
        global_shap = load_global_shap_importance()
        fig2 = go.Figure(go.Bar(
            x=global_shap["Mean_Abs_SHAP"][::-1], y=global_shap["label"][::-1], orientation="h",
            marker=dict(color=global_shap["Mean_Abs_SHAP"][::-1], colorscale=[[0, ACCENT], [1, PRIMARY]]),
        ))
        fig2.update_layout(
            template=PLOTLY_TEMPLATE, height=460, margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Mean |SHAP value|",
        )
        st.plotly_chart(fig2, width='stretch')
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Global SHAP importance file could not be loaded: {exc}")

    st.markdown('<div class="section-title">Training Information</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.markdown(
        f'<div class="card"><b>Dataset Size</b><br>{len(df):,} patient records</div>',
        unsafe_allow_html=True,
    )
    t2.markdown(
        '<div class="card"><b>Features Used</b><br>18 clinical, demographic, and appointment features</div>',
        unsafe_allow_html=True,
    )
    t3.markdown(
        f'<div class="card"><b>Class Balance</b><br>'
        f'{(1-df["no_show"].mean())*100:.1f}% Attended / {df["no_show"].mean()*100:.1f}% No-Show</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📋 Full Feature List Used by the Model"):
        st.write(", ".join(FEATURE_LABELS.values()))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main():
    render_sidebar()
    page = st.session_state.page
    if page == "Dashboard":
        page_dashboard()
    elif page == "Patient Prediction":
        page_prediction_form()
    elif page == "Prediction Result":
        page_result()
    elif page == "Explainability":
        page_explainability()
    elif page == "About Model":
        page_about_model()


if __name__ == "__main__":
    main()
