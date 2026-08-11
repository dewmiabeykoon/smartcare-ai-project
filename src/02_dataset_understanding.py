#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np

# In[15]:


df = pd.read_csv("../Dataset/smartcare_ai_dataset_1000.csv")  # relative path: run this script from the Notebook/ folder

# In[16]:


df.head()

# In[17]:


df.shape

# In[18]:


df.info()

# In[19]:


df.describe()

# In[20]:


df.columns

# In[21]:


print(df["no_show"].head())

# In[22]:


df.isnull().sum()

# In[23]:


df.duplicated().sum()

# In[24]:


df.dtypes

# In[25]:


df.nunique()

# In[ ]:



