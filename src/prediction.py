"""
Loads the trained pipeline (StandardScaler + LogisticRegression) and runs
predictions. The pipeline was fit end-to-end, so raw numeric/encoded
features go straight in - no separate scaler file to load.
"""

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.utils import MODEL_PATH, TRAIN_TEST_SPLIT_PATH


class ModelLoadError(Exception):
    pass


class PredictionError(Exception):
    pass


@st.cache_resource(show_spinner=False)
def load_model():
    if not Path(MODEL_PATH).exists():
        raise ModelLoadError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Please make sure 'logistic_regression.joblib' is present in the models/ folder."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load the trained model: {exc}") from exc
    return model


@st.cache_resource(show_spinner=False)
def load_train_reference():
    """Loads the saved train/test split - used as the SHAP background
    reference and for dashboard statistics context."""
    if not Path(TRAIN_TEST_SPLIT_PATH).exists():
        raise ModelLoadError(
            f"Training reference file not found at '{TRAIN_TEST_SPLIT_PATH}'."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = joblib.load(TRAIN_TEST_SPLIT_PATH)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Could not load training reference data: {exc}") from exc
    return data


def get_model_status() -> dict:
    """Returns a dict describing whether the model/pipeline loaded cleanly,
    without raising - used to render the dashboard status badge."""
    try:
        model = load_model()
        load_train_reference()
        n_features = getattr(model.named_steps.get("scaler", None), "n_features_in_", None)
        return {
            "ok": True,
            "message": "Model loaded successfully",
            "n_features": n_features,
        }
    except ModelLoadError as exc:
        return {"ok": False, "message": str(exc), "n_features": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": f"Unexpected error loading model: {exc}", "n_features": None}


def predict(feature_row: pd.DataFrame) -> dict:
    """
    feature_row: single-row DataFrame in the exact column order the model
    expects (see src.preprocessing.build_feature_row).

    Returns dict with class prediction, and probability for both classes.
    """
    try:
        model = load_model()
    except ModelLoadError as exc:
        raise PredictionError(str(exc)) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = model.predict_proba(feature_row)[0]
            pred_class = int(model.predict(feature_row)[0])
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(
            "The model could not generate a prediction from the provided "
            f"input. Details: {exc}"
        ) from exc

    return {
        "prediction": pred_class,          # 0 = attended, 1 = no-show
        "attended_probability": float(proba[0]),
        "no_show_probability": float(proba[1]),
    }
