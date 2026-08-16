# 🏥 SmartCare AI
## Explainable AI for Healthcare Appointment No-Show Prediction

SmartCare AI is a group-based Artificial Intelligence and Machine Learning project developed to predict whether a patient is likely to **attend or miss a scheduled healthcare appointment**.

The project implements an end-to-end machine learning workflow, including data understanding, preprocessing, exploratory data analysis, feature engineering, model development, hyperparameter tuning, model evaluation, and Explainable AI using SHAP.

---

## 📌 Project Overview

Missed medical appointments can negatively affect healthcare resource utilization, appointment scheduling, and service efficiency.

SmartCare AI aims to address this problem by developing machine learning models that learn patterns from historical healthcare appointment data and predict the likelihood of a patient being a **no-show**.

In addition to prediction, the project focuses on **Explainable AI (XAI)** to understand why a model makes a particular prediction.

---

## 🎯 Objectives

- Understand and analyse a healthcare appointment dataset.
- Clean and preprocess the dataset for machine learning.
- Perform meaningful feature engineering.
- Explore relationships between patient and appointment characteristics.
- Develop and compare multiple classification models.
- Perform hyperparameter tuning using GridSearchCV.
- Evaluate models using appropriate classification metrics.
- Select the best-performing model.
- Apply SHAP for Explainable AI.
- Provide global and local explanations for model predictions.

---

## 🧠 Machine Learning Problem

### Prediction Task

The system performs a **binary classification task**.

### Target Variable

`no_show`

| Value | Meaning |
|---|---|
| `0` | Patient attended the appointment |
| `1` | Patient did not attend the appointment |

---

## 🔄 Project Workflow

```text
Raw Dataset
     │
     ▼
Dataset Understanding
     │
     ▼
Data Preprocessing
& Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Train / Test Split
(Stratified)
     │
     ▼
Model Development
     │
     ├── Logistic Regression
     ├── Decision Tree
     ├── Random Forest
     ├── K-Nearest Neighbors
     └── XGBoost
     │
     ▼
Hyperparameter Tuning
+ Cross-Validation
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
SHAP Explainable AI
```

---

## 📊 Dataset

The project uses a healthcare appointment dataset containing patient, appointment, medical, laboratory, admission, and financial-related information.

The dataset contains **1,000 records** and multiple attributes used to identify patterns associated with appointment attendance.

### Example Feature Categories

- Patient information
- Appointment information
- Previous appointment history
- Blood pressure measurements
- Laboratory information
- Admission information
- Financial information
- Previous appointment behaviour

---

## 🧹 Data Preprocessing

The preprocessing stage includes:

- Missing value handling
- Duplicate detection and removal
- Outlier detection using the IQR method
- Feature engineering
- Categorical feature encoding
- Feature scaling
- Feature selection

Examples of engineered information include:

- Age groups
- High blood pressure indicators
- Previous appointment behaviour
- Previous missed appointment information

---

## 🔍 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Dataset distributions
- Target class distribution
- Numerical feature relationships
- Categorical feature patterns
- Correlations
- Potential outliers
- Relationships between appointment behaviour and the target variable

---

## 🤖 Machine Learning Models

Five classification algorithms were developed and compared.

### 1. Logistic Regression
An interpretable linear classification model used as a baseline and for understanding feature relationships.

### 2. Decision Tree
A tree-based model capable of learning non-linear decision boundaries.

### 3. Random Forest
An ensemble method combining multiple decision trees to improve robustness and predictive performance.

### 4. K-Nearest Neighbors (KNN)
A distance-based algorithm that predicts a class based on neighbouring observations.

### 5. XGBoost
A gradient boosting algorithm used to evaluate an advanced ensemble learning approach.

---

## ⚙️ Model Training & Hyperparameter Tuning

The dataset was divided into training and testing subsets using a stratified split:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Why `stratify=y`?

Stratification helps maintain a similar distribution of target classes in both the training and testing datasets.

### GridSearchCV

`GridSearchCV` was used to search for suitable hyperparameter combinations.

The tuning process uses:

**5-Fold Cross-Validation**

This allows different parameter combinations to be evaluated across multiple validation folds.

---

## 📈 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curves
- Training vs Testing Performance
- Overfitting Analysis

---

## 🏆 Best Performing Model

Based on the current test-set evaluation, **Logistic Regression** achieved the strongest overall performance among the evaluated models.

| Metric | Score |
|---|---:|
| Accuracy | **64.00%** |
| F1-Score | **0.6364** |
| ROC-AUC | **0.6543** |

The Logistic Regression model was selected as the best-performing model for the current experimental setup.

> **Note:** These results are based on the current dataset and experimental configuration and should not be interpreted as clinical performance.

---

## 🔎 Explainable AI (SHAP)

SmartCare AI uses **SHAP (SHapley Additive exPlanations)** to understand how individual features contribute to model predictions.

SHAP helps answer:

> **Why did the model make this prediction?**

### 🌍 Global Explainability

Global SHAP analysis identifies features with the greatest overall influence on predictions.

Important features identified through the analysis include:

- `waiting_days`
- `missed_previous_appointments`
- `previous_appointments`
- `admitted`
- `lab_charge_lkr`
- `total_bill_lkr`
- `lab_tests_count`

### 👤 Local Explainability

Local SHAP explanations are used to explain an individual patient's prediction.

A SHAP waterfall plot shows how individual feature values contribute toward a particular prediction.

```text
Patient Information
        │
        ▼
Feature Contributions
        │
        ▼
SHAP Values
        │
        ▼
Model Prediction
        │
        ▼
Appointment Attendance / No-Show
```

---

## 📁 Repository Structure

```text
smartcare-ai-project/
│
├── Data/
│   ├── smartcare_ai_dataset_1000.csv
│   ├── clean_dataset.csv        # cleaned/encoded dataset used for dashboard statistics
│   ├── preprocessed_train_test.jolib
├── Notebook/
│   ├── TASK_02_DatasetUnderstanding.ipynb
│   ├── TASK-03_Data Preprocessing.ipynb
│   ├── TASK-04_EDA.ipynb
│   ├── TASK-05_ModelDevelopment.ipynb
│   ├── TASK-06_Model EvaluationA.ipynb
│   └── TASK_07_SHAP.ipynb
│
├── asset/
│   ├── shap_bar_summary.png
│   ├── shap_beeswarm_summary.png
│   ├── shap_waterfall_sample.png
│   ├── shap_global_importance.csv
│   ├── shap_local_sample_0.csv
│   └── lr_coefficients.png
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── knn.joblib
│   ├── xgboost.joblib
│   └── hyperparameter_summary.csv
│
│
├── src/
│   ├── preprocessing.py         # raw form input -> exact feature row the model expects
│   ├── prediction.py            # cached model loading + safe predict wrapper
│   ├── explainability.py        # live SHAP (LinearExplainer) + global importance
│   ├── utils.py                 # category mappings, risk thresholds, CSS theme
│  
├── app.py                     # Streamlit UI: navigation, pages, charts
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost |
| Explainable AI | SHAP |
| Visualization | Matplotlib |
| Model Saving | Joblib |
| Development | Jupyter Notebook / Google Colab |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dewmiabeykoon/smartcare-ai-project.git
```

### 2. Navigate to the Project

```bash
cd smartcare-ai-project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the notebooks in the following order:

```text
1. Dataset Understanding
        ↓
2. Data Preprocessing
        ↓
3. Exploratory Data Analysis
        ↓
4. Model Development
        ↓
5. Model Evaluation
        ↓
6. Explainable AI using SHAP
```

The corresponding Python source files are available inside the `src/` directory.

---

## 🔐 Data Leakage Prevention

Data leakage prevention was considered throughout the machine learning workflow.

The dataset was split before final model evaluation:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Cross-validation is used during hyperparameter tuning so that model selection does not rely on the final test set.

---

## 📚 Coursework Tasks

| Task | Description |
|---|---|
| Task 02 | Dataset Understanding |
| Task 03 | Data Preprocessing & Feature Engineering |
| Task 04 | Exploratory Data Analysis |
| Task 05 | Model Development |
| Task 06 | Model Evaluation |
| Task 07 | Explainable AI using SHAP |

---

## 👥 Group Members

This project was collaboratively developed by:

- **Dewmi Abeykoon**
- **Jayani Wathsala**
- **Maliksha Pawani**
- **Pabasara Maitipe**

All members contributed to the development, analysis, evaluation, and documentation of the project as part of the group coursework.

---

## 🎓 Academic Context

This project was developed as part of an **Artificial Intelligence coursework project** and demonstrates practical application of:

- Artificial Intelligence
- Machine Learning
- Data Preprocessing
- Exploratory Data Analysis
- Classification
- Hyperparameter Optimization
- Model Evaluation
- Explainable AI

---

## ⚠️ Disclaimer

SmartCare AI is an **academic project** developed for educational purposes.

The predictions generated by this system should **not be used as a substitute for professional medical advice, diagnosis, or clinical decision-making**.

The model is trained using a limited dataset and may not generalize to real-world healthcare environments.

---

## 🚀 Future Improvements

- Increase the dataset size
- Test additional machine learning algorithms
- Improve class imbalance handling
- Perform advanced feature engineering
- Apply model calibration
- Develop a complete web-based prediction interface
- Deploy the model through an API
- Add real-time prediction capabilities
- Perform external validation
- Implement continuous model monitoring

---

## ⭐ Key Highlights

```text
✔ End-to-End Machine Learning Workflow
✔ Healthcare Appointment Prediction
✔ 5 Classification Models
✔ Hyperparameter Tuning
✔ 5-Fold Cross-Validation
✔ Stratified Train/Test Split
✔ Data Preprocessing & Feature Engineering
✔ Comprehensive Model Evaluation
✔ SHAP Explainable AI
✔ Global Feature Importance
✔ Local Prediction Explanation
✔ Saved Machine Learning Models
✔ Collaborative GitHub Project
```

---

## 🔗 Repository

GitHub: https://github.com/dewmiabeykoon/smartcare-ai-project
