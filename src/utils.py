"""
Shared constants, category mappings, and styling helpers for the
SmartCare AI No-Show Prediction dashboard.

These category-to-code mappings mirror exactly what was used when the
training dataset was label-encoded (see Notebook/TASK-03_Data
Preprocessing.ipynb). They must stay in sync with the trained model.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
TRAIN_TEST_SPLIT_PATH = MODEL_DIR / "train_test_split.joblib"
EVAL_RESULTS_PATH = MODEL_DIR / "model_evaluation_results.csv"
SHAP_GLOBAL_PATH = MODEL_DIR / "shap_global_importance.csv"
CLEAN_DATASET_PATH = DATA_DIR / "clean_dataset.csv"

# ---------------------------------------------------------------------------
# Categorical encodings (LabelEncoder, fitted alphabetically on the raw
# training data). Order matters - do not resort.
# ---------------------------------------------------------------------------
GENDER_MAP = {"Female": 0, "Male": 1}

BLOOD_GROUP_MAP = {
    "A+": 0, "A-": 1, "AB+": 2, "AB-": 3,
    "B+": 4, "B-": 5, "O+": 6, "O-": 7,
}

DEPARTMENT_MAP = {
    "Cardiology": 0,
    "General Medicine": 1,
    "Laboratory Services": 2,
    "Neurology": 3,
    "Orthopedics": 4,
    "Pediatrics": 5,
    "Radiology": 6,
}

DIAGNOSIS_MAP = {
    "Asthma": 0,
    "Back Pain": 1,
    "Chest Pain": 2,
    "Diabetes": 3,
    "Fever": 4,
    "Fracture": 5,
    "Hypertension": 6,
    "Kidney Infection": 7,
    "Migraine": 8,
    "Pneumonia": 9,
}

# Reverse maps, used for chart labels
DEPARTMENT_NAMES = {v: k for k, v in DEPARTMENT_MAP.items()}
DIAGNOSIS_NAMES = {v: k for k, v in DIAGNOSIS_MAP.items()}
BLOOD_GROUP_NAMES = {v: k for k, v in BLOOD_GROUP_MAP.items()}
GENDER_NAMES = {v: k for k, v in GENDER_MAP.items()}

# Final feature order expected by the trained pipeline
FEATURE_ORDER = [
    "age", "gender", "blood_group", "department", "diagnosis",
    "waiting_days", "previous_appointments", "missed_previous_appointments",
    "previous_admissions", "systolic_bp", "diastolic_bp",
    "blood_sugar_mg_dl", "cholesterol_mg_dl", "bmi", "consultation_fee_lkr",
    "Age_Group", "High_BP", "Missed_Rate",
]

# Human-friendly labels for SHAP / importance charts
FEATURE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "blood_group": "Blood Group",
    "department": "Department",
    "diagnosis": "Diagnosis",
    "waiting_days": "Waiting Days",
    "previous_appointments": "Previous Appointments",
    "missed_previous_appointments": "Previously Missed Appointments",
    "previous_admissions": "Previous Admissions",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "blood_sugar_mg_dl": "Blood Sugar (mg/dL)",
    "cholesterol_mg_dl": "Cholesterol (mg/dL)",
    "bmi": "BMI",
    "consultation_fee_lkr": "Consultation Fee (LKR)",
    "Age_Group": "Age Group",
    "High_BP": "High Blood Pressure Flag",
    "Missed_Rate": "Historical Missed-Appointment Rate",
}

# ---------------------------------------------------------------------------
# Risk thresholds - configurable, not silently hard-coded.
# Defaults were chosen around the model's natural probability spread on the
# balanced training set (~49.5% base no-show rate). Adjustable at runtime
# from the sidebar; session_state overrides these defaults.
# ---------------------------------------------------------------------------
DEFAULT_RISK_THRESHOLDS = {"low": 0.40, "high": 0.65}


def get_risk_level(no_show_probability: float, thresholds: dict) -> str:
    """Convert a no-show probability into a LOW / MEDIUM / HIGH risk label."""
    if no_show_probability < thresholds["low"]:
        return "LOW RISK"
    if no_show_probability < thresholds["high"]:
        return "MEDIUM RISK"
    return "HIGH RISK"


RISK_COLORS = {
    "LOW RISK": "#16a34a",
    "MEDIUM RISK": "#d97706",
    "HIGH RISK": "#dc2626",
}

RECOMMENDATIONS = {
    "LOW RISK": [
        "Send the standard automated appointment reminder.",
        "No additional follow-up action required at this time.",
    ],
    "MEDIUM RISK": [
        "Send an additional reminder 48 hours before the appointment.",
        "Request confirmation from the patient ahead of the scheduled date.",
    ],
    "HIGH RISK": [
        "Send an early reminder well ahead of the appointment date.",
        "Send a confirmation request and ask the patient to explicitly confirm.",
        "Consider an additional follow-up reminder or a courtesy phone call.",
    ],
}

# ---------------------------------------------------------------------------
# Theme - dark healthcare AI palette
# ---------------------------------------------------------------------------
PRIMARY = "#2dd4bf"       # bright teal - accent on dark bg
PRIMARY_DARK = "#0d9488"
ACCENT = "#60a5fa"        # soft blue accent
BG = "#0b1220"            # deep navy background
BG_GRADIENT_END = "#0f1b2d"
CARD_BG = "#131c2e"       # slightly lighter navy for cards
CARD_BORDER = "rgba(255,255,255,0.08)"
TEXT_DARK = "#e5e7eb"     # near-white primary text
TEXT_MUTED = "#94a3b8"
DANGER = "#f87171"
WARNING = "#fbbf24"
SUCCESS = "#4ade80"

PLOTLY_TEMPLATE = "plotly_dark"


def load_custom_css() -> str:
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #MainMenu, footer, header {{visibility: hidden;}}

        .stApp {{
            background: linear-gradient(180deg, {BG} 0%, {BG_GRADIENT_END} 100%);
            color: {TEXT_DARK};
        }}

        /* Headings and body text */
        h1, h2, h3, h4, h5, h6, p, label, span, li {{
            color: {TEXT_DARK};
        }}
        .stMarkdown, .stCaption, .stText {{
            color: {TEXT_DARK};
        }}
        [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTED} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0a1a1c 0%, {PRIMARY_DARK} 100%);
            border-right: 1px solid {CARD_BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: #e6fffa !important;
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            font-weight: 600;
        }}
        section[data-testid="stSidebar"] [data-baseweb="radio"] {{
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            padding: .3rem .5rem;
            margin-bottom: .15rem;
        }}

        /* Hero */
        .hero {{
            background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {ACCENT} 100%);
            padding: 2.4rem 2.2rem;
            border-radius: 20px;
            color: white;
            box-shadow: 0 10px 34px rgba(0, 0, 0, 0.45);
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .hero h1 {{
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 .4rem 0;
            color: white !important;
        }}
        .hero p {{
            font-size: 1.02rem;
            opacity: .92;
            margin: 0;
            max-width: 720px;
            color: white !important;
        }}

        /* Top pill navigation (sits right under the hero) - scoped so it
           doesn't affect any other buttons in the app */
        div[class*="st-key-topnav_bar"] {{
            margin-bottom: 1.4rem;
        }}
        div[class*="st-key-topnav_bar"] button {{
            width: 100%;
            border-radius: 999px !important;
            font-weight: 700 !important;
            box-shadow: none !important;
        }}
        div[class*="st-key-topnav_bar"] button[kind="secondary"] {{
            background: {CARD_BG} !important;
            color: {TEXT_MUTED} !important;
            border: 1px solid {CARD_BORDER} !important;
        }}
        div[class*="st-key-topnav_bar"] button[kind="secondary"]:hover {{
            border-color: {PRIMARY} !important;
            color: {PRIMARY} !important;
            transform: translateY(-1px);
        }}
        div[class*="st-key-topnav_bar"] button[kind="primary"] {{
            background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%) !important;
            color: #052e2b !important;
            border: none !important;
            box-shadow: 0 6px 18px rgba(45, 212, 191, 0.35) !important;
        }}

        /* Cards */
        .card {{
            background: {CARD_BG};
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            border: 1px solid {CARD_BORDER};
            transition: transform .15s ease, box-shadow .15s ease;
            margin-bottom: 1rem;
            color: {TEXT_DARK};
        }}
        .card b {{ color: {TEXT_DARK}; }}
        .card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
            border-color: rgba(45, 212, 191, 0.3);
        }}

        .stat-card {{
            background: {CARD_BG};
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            border: 1px solid {CARD_BORDER};
            border-left: 4px solid {PRIMARY};
            transition: transform .15s ease;
        }}
        .stat-card:hover {{ transform: translateY(-3px); }}
        .stat-label {{
            color: {TEXT_MUTED} !important;
            font-size: .82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .04em;
        }}
        .stat-value {{
            color: {TEXT_DARK} !important;
            font-size: 1.9rem;
            font-weight: 800;
            margin-top: .15rem;
        }}

        .section-title {{
            font-size: 1.3rem;
            font-weight: 700;
            color: {TEXT_DARK} !important;
            margin: 1.4rem 0 .7rem 0;
            border-left: 5px solid {PRIMARY};
            padding-left: .6rem;
        }}

        .badge {{
            display: inline-block;
            padding: .3rem .8rem;
            border-radius: 999px;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .03em;
        }}
        .badge-ok {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; }}
        .badge-warn {{ background: rgba(251, 191, 36, 0.15); color: #fbbf24; }}
        .badge-err {{ background: rgba(248, 113, 113, 0.15); color: #f87171; }}

        /* Result card */
        .result-card {{
            border-radius: 20px;
            padding: 2rem 2.2rem;
            color: white;
            box-shadow: 0 14px 34px rgba(0,0,0,0.5);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .result-card h2 {{
            margin: 0 0 .3rem 0;
            font-size: 1.7rem;
            font-weight: 800;
            color: white !important;
        }}
        .result-card p {{
            margin: 0;
            opacity: .95;
            color: white !important;
        }}

        .risk-pill {{
            display: inline-block;
            padding: .45rem 1.1rem;
            border-radius: 999px;
            font-weight: 800;
            font-size: .95rem;
            letter-spacing: .04em;
            background: rgba(255,255,255,0.18);
            margin-top: .6rem;
        }}

        /* Form group headers */
        .form-group-title {{
            font-weight: 700;
            color: {PRIMARY} !important;
            font-size: 1.02rem;
            margin: 1.1rem 0 .3rem 0;
            padding-bottom: .3rem;
            border-bottom: 2px solid {CARD_BORDER};
        }}

        /* Default (non-nav) buttons - e.g. form submit, quick access */
        div.stButton > button,
        .stFormSubmitButton > button {{
            background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {ACCENT} 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: .6rem 1.6rem;
            font-weight: 700;
            transition: transform .12s ease, box-shadow .12s ease;
            box-shadow: 0 6px 18px rgba(96, 165, 250, 0.25);
        }}
        div.stButton > button:hover, .stFormSubmitButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(96, 165, 250, 0.35);
        }}

        .explain-box {{
            background: rgba(45, 212, 191, 0.08);
            border-left: 5px solid {PRIMARY};
            border-radius: 12px;
            padding: 1rem 1.2rem;
            color: {TEXT_DARK};
            line-height: 1.55;
        }}

        .rec-item {{
            background: {CARD_BG};
            border-radius: 12px;
            padding: .7rem 1rem;
            margin-bottom: .5rem;
            border: 1px solid {CARD_BORDER};
            font-weight: 500;
            color: {TEXT_DARK};
        }}

        /* Form widgets - inputs, selects */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
        .stTextArea textarea {{
            background-color: {CARD_BG} !important;
            color: {TEXT_DARK} !important;
            border: 1px solid {CARD_BORDER} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox svg {{ fill: {TEXT_MUTED} !important; }}
        div[data-baseweb="popover"] li {{
            background-color: {CARD_BG} !important;
            color: {TEXT_DARK} !important;
        }}
        .stForm {{
            background: rgba(255,255,255,0.02);
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            padding: 1.4rem 1.6rem;
        }}

        /* Expander */
        .streamlit-expanderHeader, details summary {{
            background: {CARD_BG} !important;
            color: {TEXT_DARK} !important;
            border-radius: 10px !important;
        }}
        details {{
            border: 1px solid {CARD_BORDER} !important;
            border-radius: 10px !important;
        }}

        /* Dataframe / table */
        [data-testid="stDataFrame"] {{
            background: {CARD_BG};
            border-radius: 12px;
            border: 1px solid {CARD_BORDER};
        }}

        /* Slider */
        .stSlider [data-baseweb="slider"] {{
            color: {PRIMARY};
        }}

        /* Alerts */
        div[data-baseweb="notification"], .stAlert {{
            border-radius: 12px !important;
        }}
    </style>
    """
