import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

shap.initjs()

# Mount google Drive

SAVE_DIR = "../models"  # relative path: run this script from the Notebook/ folder

split_data = joblib.load(f"{SAVE_DIR}/train_test_split.joblib")
X_train = split_data["X_train"]
X_test  = split_data["X_test"]
y_test  = split_data["y_test"]

best_model_name = "Logistic Regression"

best_model = joblib.load(f"{SAVE_DIR}/logistic_regression.joblib")

print(f"Explaining model: {best_model_name}")
print("X_test shape:", X_test.shape)

explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

print("SHAP values shape:", shap_values.values.shape)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Global Feature Importance - Mean |SHAP Value|", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_bar_summary.png", dpi=150, bbox_inches="tight")
plt.show()

#Beeswarm Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm_summary.png", dpi=150, bbox_inches="tight")
plt.show()

sample_idx = 0

plt.figure()
shap.plots.waterfall(shap_values[sample_idx], show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Actual label for this patient: {'No-Show' if y_test.iloc[sample_idx]==1 else 'Attended'}")
print(f"Predicted label: {'No-Show' if best_model.predict(X_test.iloc[[sample_idx]])[0]==1 else 'Attended'}")

coef_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": best_model.coef_[0]
}).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)

plt.figure(figsize=(9, 7))
colors = ["#d62728" if c > 0 else "#1f77b4" for c in coef_df["Coefficient"][:10]]
plt.barh(coef_df["Feature"][:10][::-1], coef_df["Coefficient"][:10][::-1], color=colors[::-1])
plt.title("Top 10 Logistic Regression Coefficients\n(Red = increases No-Show risk, Blue = decreases)", fontweight="bold")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig("lr_coefficients.png", dpi=150, bbox_inches="tight")
plt.show()

coef_df.head(10)

coef_df.to_csv(f"{SAVE_DIR}/shap_feature_importance.csv", index=False)
print(f"Saved: {SAVE_DIR}/shap_feature_importance.csv")
