#!/usr/bin/env python
# coding: utf-8

# # 01. Import Libraries

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # 02 Load Clean Dataset

# In[7]:



# In[8]:


file_path = "../Dataset/clean_dataset.csv"  # relative path: run this script from the Notebook/ folder
df = pd.read_csv(file_path)
df.head()

# In[9]:


df.describe()

# # 03. Exploratory Data Analysis (EDA)
# 

# ### Univariate Analysis (Single Variable Analysis)

# In[10]:


numeric_cols = [
    "age",
    "waiting_days",
    "bmi",
    "blood_sugar_mg_dl",
    "cholesterol_mg_dl",
    "total_bill_lkr"
]

df[numeric_cols].hist(figsize=(15,10), bins=20)

plt.tight_layout()
plt.show()

# ### Bivariate & Categorical Analysis

# In[11]:


plt.figure(figsize=(15,6))

sns.boxplot(data=df[numeric_cols])

plt.xticks(rotation=45)

plt.show()

# In[12]:


plt.figure(figsize=(15,10))

corr = df.select_dtypes(include=['int64','float64']).corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")

plt.show()

# ## Scatter plot

# In[13]:


plt.figure(figsize=(8,6))

sns.scatterplot(
    x="age",
    y="total_bill_lkr",
    data=df
)

plt.show()

# In[14]:


plt.figure(figsize=(8,6))

sns.scatterplot(
    x="bmi",
    y="blood_sugar_mg_dl",
    data=df
)

plt.show()

# ### Categorical Feature Counts

# In[15]:


plt.figure(figsize=(6,5))

sns.countplot(x="no_show", data=df)

plt.title("Appointment No Show Distribution")

plt.show()

# In[16]:


plt.figure(figsize=(10,5))

sns.countplot(x="department", data=df)

plt.xticks(rotation=45)

plt.show()

# In[17]:


plt.figure(figsize=(5,5))

sns.countplot(x="gender", data=df)

plt.show()

# In[18]:


plt.figure(figsize=(12,5))

sns.countplot(y="diagnosis", data=df)

plt.show()

# In[19]:


plt.figure(figsize=(8,6))

sns.boxplot(
    x="no_show",
    y="waiting_days",
    data=df
)

plt.show()

# In[20]:


plt.figure(figsize=(8,6))

sns.boxplot(
    x="no_show",
    y="bmi",
    data=df
)

plt.show()

# In[21]:


plt.figure(figsize=(8,6))

sns.boxplot(
    x="no_show",
    y="previous_appointments",
    data=df
)

plt.show()

# ### Pairplot Analysis

# In[22]:


sns.pairplot(
    df[
        [
            "age",
            "bmi",
            "waiting_days",
            "blood_sugar_mg_dl",
            "no_show"
        ]
    ],
    hue="no_show"
)

plt.show()
