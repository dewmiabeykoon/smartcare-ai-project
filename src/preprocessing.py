"""
Turns raw, human-readable patient form input into the exact 18-column
feature row the trained pipeline was fit on.

The trained model (models/logistic_regression.joblib) is a scikit-learn
Pipeline([StandardScaler, LogisticRegression]). Scaling is therefore
handled internally by the pipeline itself - this module only has to
reproduce the label-encoding / feature-engineering steps that were done
before the pipeline was fit (see Notebook/TASK-03_Data Preprocessing.ipynb).

Three engineered features are derived automatically rather than asked of
the user a second time, because they are deterministic functions of other
inputs already on the form:
    - Age_Group   -> binned from `age`
    - High_BP     -> flag derived from systolic/diastolic BP
    - Missed_Rate -> missed_previous_appointments / previous_appointments
"""

import pandas as pd

from src.utils import (
    FEATURE_ORDER,
    GENDER_MAP,
    BLOOD_GROUP_MAP,
    DEPARTMENT_MAP,
    DIAGNOSIS_MAP,
)


def compute_age_group(age: int) -> int:
    """Reproduces the age-bin encoding used during training."""
    if age <= 18:
        return 1
    if age <= 35:
        return 4
    if age <= 50:
        return 0
    if age <= 65:
        return 2
    return 3


def compute_high_bp(systolic_bp: int, diastolic_bp: int) -> int:
    """Hypertension flag: systolic >= 140 or diastolic >= 90."""
    return int(systolic_bp >= 140 or diastolic_bp >= 90)


def compute_missed_rate(missed_previous: int, previous_appointments: int) -> float:
    if previous_appointments <= 0:
        return 0.0
    return round(missed_previous / previous_appointments, 4)


class InputValidationError(Exception):
    pass


def validate_raw_input(raw: dict) -> list:
    """Returns a list of human-readable validation error messages (empty if OK)."""
    errors = []

    def _check_range(key, low, high, label):
        val = raw.get(key)
        if val is None:
            errors.append(f"{label} is required.")
        elif not (low <= val <= high):
            errors.append(f"{label} must be between {low} and {high}.")

    _check_range("age", 0, 120, "Age")
    _check_range("waiting_days", 0, 365, "Waiting days")
    _check_range("previous_appointments", 0, 200, "Previous appointments")
    _check_range("missed_previous_appointments", 0, 200, "Missed previous appointments")
    _check_range("previous_admissions", 0, 100, "Previous admissions")
    _check_range("systolic_bp", 60, 260, "Systolic BP")
    _check_range("diastolic_bp", 30, 180, "Diastolic BP")
    _check_range("blood_sugar_mg_dl", 40, 600, "Blood sugar")
    _check_range("cholesterol_mg_dl", 80, 500, "Cholesterol")
    _check_range("bmi", 8, 70, "BMI")
    _check_range("consultation_fee_lkr", 0, 100000, "Consultation fee")

    if raw.get("missed_previous_appointments") is not None and raw.get("previous_appointments") is not None:
        if raw["missed_previous_appointments"] > raw["previous_appointments"]:
            errors.append(
                "Missed previous appointments cannot exceed total previous appointments."
            )

    for key, mapping, label in [
        ("gender", GENDER_MAP, "Gender"),
        ("blood_group", BLOOD_GROUP_MAP, "Blood group"),
        ("department", DEPARTMENT_MAP, "Department"),
        ("diagnosis", DIAGNOSIS_MAP, "Diagnosis"),
    ]:
        if raw.get(key) not in mapping:
            errors.append(f"{label} selection is invalid.")

    return errors


def build_feature_row(raw: dict) -> pd.DataFrame:
    """
    raw: dict of human-readable form values, e.g.
        {
            "age": 45, "gender": "Male", "blood_group": "A+",
            "department": "Cardiology", "diagnosis": "Hypertension",
            "waiting_days": 5, "previous_appointments": 4,
            "missed_previous_appointments": 1, "previous_admissions": 0,
            "systolic_bp": 132, "diastolic_bp": 84,
            "blood_sugar_mg_dl": 110, "cholesterol_mg_dl": 190,
            "bmi": 24.5, "consultation_fee_lkr": 2000,
        }

    Returns a single-row DataFrame with columns in FEATURE_ORDER, ready to
    be passed straight into the trained pipeline's .predict()/.predict_proba().
    """
    errors = validate_raw_input(raw)
    if errors:
        raise InputValidationError("; ".join(errors))

    age_group = compute_age_group(raw["age"])
    high_bp = compute_high_bp(raw["systolic_bp"], raw["diastolic_bp"])
    missed_rate = compute_missed_rate(
        raw["missed_previous_appointments"], raw["previous_appointments"]
    )

    row = {
        "age": raw["age"],
        "gender": GENDER_MAP[raw["gender"]],
        "blood_group": BLOOD_GROUP_MAP[raw["blood_group"]],
        "department": DEPARTMENT_MAP[raw["department"]],
        "diagnosis": DIAGNOSIS_MAP[raw["diagnosis"]],
        "waiting_days": raw["waiting_days"],
        "previous_appointments": raw["previous_appointments"],
        "missed_previous_appointments": raw["missed_previous_appointments"],
        "previous_admissions": raw["previous_admissions"],
        "systolic_bp": raw["systolic_bp"],
        "diastolic_bp": raw["diastolic_bp"],
        "blood_sugar_mg_dl": raw["blood_sugar_mg_dl"],
        "cholesterol_mg_dl": raw["cholesterol_mg_dl"],
        "bmi": raw["bmi"],
        "consultation_fee_lkr": raw["consultation_fee_lkr"],
        "Age_Group": age_group,
        "High_BP": high_bp,
        "Missed_Rate": missed_rate,
    }

    df = pd.DataFrame([row])[FEATURE_ORDER]
    return df
