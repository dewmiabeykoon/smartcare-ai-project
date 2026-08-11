import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)


SAVE_DIR = "../models"  # relative path: run this script from the Notebook/ folder

split_data = joblib.load(f"{SAVE_DIR}/train_test_split.joblib")
X_train = split_data["X_train"]
X_test  = split_data["X_test"]
y_train = split_data["y_train"]
y_test  = split_data["y_test"]

# Load all trained models
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "KNN": "knn.joblib",
    "XGBoost": "xgboost.joblib"
}

models = {name: joblib.load(f"{SAVE_DIR}/{file}") for name, file in model_files.items()}
print("Loaded models:", list(models.keys()))

predictions = {}
probabilities = {}

for name, model in models.items():
    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]  # probability of class 1 (No-Show)

print("Predictions generated for:", list(predictions.keys()))

results = []

for name in models.keys():
    y_pred = predictions[name]
    y_prob = probabilities[name]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    })

comparison_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
comparison_df

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

plot_df = comparison_df.set_index("Model")[metrics_to_plot]

ax = plot_df.plot(kind="bar", figsize=(12, 6), colormap="viridis")
plt.title("Model Performance Comparison — Appointment No-Show Prediction", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    cm = confusion_matrix(y_test, predictions[name])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Attended", "No-Show"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(name, fontweight="bold")

# hide unused subplot(s)
for j in range(len(models), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 7))

for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_score = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

best_model_name = comparison_df.iloc[0]["Model"]
best_model = models[best_model_name]

print(f" Best Performing Model: {best_model_name}\n")
print("Overall Metrics:")
print(comparison_df.iloc[0])

print("\nDetailed Classification Report:")
print(classification_report(y_test,
                            predictions[best_model_name],
                            target_names=["Attended", "No-Show"]))

comparison_df.to_csv(f"{SAVE_DIR}/model_evaluation_results.csv", index=False)

print(f"Saved evaluation results to: {SAVE_DIR}/model_evaluation_results.csv")
print(f"\nBest model selected: {best_model_name}")
