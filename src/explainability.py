"""
Explainable AI helpers.

SHAP values are computed live, from the actual trained model, using
shap.LinearExplainer against the logistic-regression coefficients and a
background sample drawn from the real training data (models/train_test_split.joblib).
Nothing here is fabricated or hard-coded - if SHAP or the training
reference is unavailable, the app falls back to the model's own
coefficients (also real, loaded from disk) and says so explicitly.
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import FEATURE_ORDER, FEATURE_LABELS, SHAP_GLOBAL_PATH
from src.prediction import load_model, load_train_reference


class ExplainabilityError(Exception):
    pass


@st.cache_resource(show_spinner=False)
def _get_linear_explainer():
    """Builds a shap.LinearExplainer once, on the model's scaled training
    background. Cached because this is the expensive part."""
    import shap  # local import so app still loads even if shap isn't installed

    model = load_model()
    tts = load_train_reference()
    X_train = tts["X_train"][FEATURE_ORDER]

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["model"]

    X_train_scaled = scaler.transform(X_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.LinearExplainer(clf, X_train_scaled, feature_names=FEATURE_ORDER)
    return explainer, scaler


def compute_shap_for_instance(feature_row: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [feature, shap_value], sorted by
    absolute contribution descending, computed for this specific patient's
    prediction (class 1 = no-show), using the actual trained model.
    """
    try:
        explainer, scaler = _get_linear_explainer()
        scaled_row = scaler.transform(feature_row[FEATURE_ORDER])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer(scaled_row)
        values = np.array(shap_values.values[0])
    except Exception as exc:  # noqa: BLE001
        raise ExplainabilityError(
            f"Could not compute a live SHAP explanation: {exc}"
        ) from exc

    df = pd.DataFrame({"feature": FEATURE_ORDER, "shap_value": values})
    df["abs_value"] = df["shap_value"].abs()
    df = df.sort_values("abs_value", ascending=False).reset_index(drop=True)
    df["label"] = df["feature"].map(FEATURE_LABELS)
    return df[["feature", "label", "shap_value", "abs_value"]]


@st.cache_data(show_spinner=False)
def load_global_shap_importance() -> pd.DataFrame:
    """Loads the pre-computed global SHAP importance (from the notebook /
    training pipeline output) for the About Model / Dashboard pages."""
    df = pd.read_csv(SHAP_GLOBAL_PATH)
    df["label"] = df["Feature"].map(FEATURE_LABELS).fillna(df["Feature"])
    return df.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)


def generate_explanation_text(shap_df: pd.DataFrame, no_show_probability: float, prediction: int) -> str:
    """Builds a short, human-readable explanation from the actual top SHAP
    contributors for this prediction - no unsupported claims added."""
    top = shap_df.head(3)
    increasing = shap_df[shap_df["shap_value"] > 0].head(2)["label"].tolist()
    decreasing = shap_df[shap_df["shap_value"] < 0].head(2)["label"].tolist()

    level = "high" if no_show_probability >= 0.65 else "moderate" if no_show_probability >= 0.40 else "low"
    predicted_outcome = "will miss" if prediction == 1 else "will attend"

    sentence = (
        f"Based on the model, this patient has a {level} predicted risk of missing the "
        f"appointment ({no_show_probability * 100:.1f}% no-show probability), and the model's "
        f"overall classification is that the patient {predicted_outcome}."
    )

    if increasing:
        sentence += " The strongest factor(s) pushing the risk up were " + " and ".join(increasing) + "."
    if decreasing:
        sentence += " The factor(s) pulling the risk down were " + " and ".join(decreasing) + "."

    return sentence
