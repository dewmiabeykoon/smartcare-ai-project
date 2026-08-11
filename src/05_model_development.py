import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


file_path = "../Dataset/clean_dataset.csv"  # relative path: run this script from the Notebook/ folder
df = pd.read_csv(file_path)
df.head()

X = df.drop(columns=["no_show"])
y = df["no_show"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train class balance:\n", y_train.value_counts(normalize=True))
print("Test class balance:\n", y_test.value_counts(normalize=True))

models = {}
best_params = {}

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
models["Logistic Regression"] = lr

# 2. Decision Tree (tuned)
dt_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"]
}
dt_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_grid, cv=5, scoring="accuracy", n_jobs=-1)
dt_search.fit(X_train, y_train)
models["Decision Tree"] = dt_search.best_estimator_
best_params["Decision Tree"] = dt_search.best_params_

# 3. Random Forest (tuned)
rf_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
rf_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=5, scoring="accuracy", n_jobs=-1)
rf_search.fit(X_train, y_train)
models["Random Forest"] = rf_search.best_estimator_
best_params["Random Forest"] = rf_search.best_params_

# 4. KNN (tuned)
knn_grid = {"n_neighbors": [3, 5, 7, 9, 11]}
knn_search = GridSearchCV(KNeighborsClassifier(), knn_grid, cv=5, scoring="accuracy", n_jobs=-1)
knn_search.fit(X_train, y_train)
models["KNN"] = knn_search.best_estimator_
best_params["KNN"] = knn_search.best_params_

# 5. XGBoost (tuned)
if XGBOOST_AVAILABLE:
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    }
    xgb_search = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        xgb_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    xgb_search.fit(X_train, y_train)
    models["XGBoost"] = xgb_search.best_estimator_
    best_params["XGBoost"] = xgb_search.best_params_

print("Models trained:", list(models.keys()))
print("\nBest hyperparameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

results = []
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append({"Model": name, "Test Accuracy": round(acc, 4)})

results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
results_df

import os

SAVE_DIR = "../models"  # relative path: run this script from the Notebook/ folder
os.makedirs(SAVE_DIR, exist_ok=True)

# Save trained models
for name, model in models.items():
    filename = os.path.join(
        SAVE_DIR,
        f"{name.replace(' ', '_').lower()}.joblib"
    )
    joblib.dump(model, filename)
    print("Saved:", filename)

# Save train/test split
joblib.dump(
    {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X.columns.tolist()
    },
    os.path.join(SAVE_DIR, "train_test_split.joblib")
)

print("All models and data saved successfully!")
