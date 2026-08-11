import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "../Dataset/smartcare_ai_dataset_1000.csv"  # relative path: run this script from the Notebook/ folder
df = pd.read_csv(file_path)
df.head()

df.isnull().sum()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

df.loc[df["admitted"] == 0, "room_type"] = df.loc[df["admitted"] == 0, "room_type"].fillna("Not Applicable")

admitted_mode = df.loc[df["admitted"] == 1, "room_type"].mode()[0]
df.loc[df["admitted"] == 1, "room_type"] = df.loc[df["admitted"] == 1, "room_type"].fillna(admitted_mode)

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isnull().sum()

df["room_type"].value_counts()

df.duplicated().sum()

df = df.drop_duplicates()
print(df.shape)

plt.figure(figsize=(15,6))
df.boxplot(rot=90)
plt.show()

exclude_from_outlier_treatment = ["record_id", "no_show", "readmitted_30_days"]
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.difference(exclude_from_outlier_treatment)

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)

df["Age_Group"] = pd.cut(
    df["age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"]
)

df["High_BP"] = (
    (df["systolic_bp"] >= 140) |
    (df["diastolic_bp"] >= 90)
).astype(int)

df["Missed_Rate"] = (
    df["missed_previous_appointments"] /
    (df["previous_appointments"] + 1)
)

label = LabelEncoder()

cat_cols = df.select_dtypes(include=['object', 'category']).columns

for col in cat_cols:
    df[col] = label.fit_transform(df[col])

df.head()

scaler = StandardScaler()

exclude_from_scaling = ["record_id", "no_show", "readmitted_30_days"]
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.difference(exclude_from_scaling)

df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(
    columns=[
        "record_id",
        "patient_id",
        "appointment_status",
        "appointment_date",
        "no_show",
        "readmitted_30_days",
        "disease_risk_level"
    ]
)

y = df["no_show"]

print(X.shape)
print(y.shape)

clean_data = pd.concat([X, y], axis=1)
clean_data.to_csv("../Dataset/clean_dataset.csv",
    index=False,)
print("Dataset successfully saved!")

clean_data.head()

# Sanity check: confirm the target variable is still clean binary 0/1, not scaled decimals
clean_data["no_show"].value_counts()
