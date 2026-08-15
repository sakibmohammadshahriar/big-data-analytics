#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install numpy


# In[3]:


pip install datetime


# In[4]:


import pandas as pd
import numpy as np
from datetime import datetime


# In[5]:


# Load the dataset
df = pd.read_csv("uncleaned_dataset (1).csv")


# In[7]:


df


# In[8]:


# Name cleanup
df = df.dropna(subset=['Name'])  


# In[9]:


df


# In[12]:


# 1. Clean Age column
df['Age'] = pd.to_numeric(df['Age'], errors='coerce') 
df.loc[df['Age'] < 0, 'Age'] = np.nan
df['Age'] = df['Age'].fillna(df['Age'].median())       


# In[13]:


df


# In[14]:


# 2. Standardize Gender
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
df['Gender'] = df['Gender'].fillna('Unknown')


# In[15]:


df


# In[16]:


# 3. Clean Income
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
df.loc[df['Income'] < 0, 'Income'] = np.nan
df['Income'] = df['Income'].fillna(df['Income'].median())


# In[17]:


df


# In[18]:


# 4. Fix JoinDate
df['JoinDate'] = pd.to_datetime(df['JoinDate'], errors='coerce')
df['JoinDate'] = df['JoinDate'].fillna(method='ffill')


# In[19]:


df


# In[20]:


# 5. Clean Subscription column
df['Subscription'] = df['Subscription'].replace({
    'Y': 'Yes', 'N': 'No', '1': 'Yes', '0': 'No'
})
df['Subscription'] = df['Subscription'].fillna('No')


# In[21]:


df


# In[22]:


# 6. Clean Email (fill with placeholder)
df['Email'] = df['Email'].fillna('unknown@example.com')


# In[ ]:


# 6. Clean Email (fill with placeholder)
df['Email'] = df['Email'].fillna('unknown@example.com')


# In[23]:


df


# In[24]:


# 7. Clean Country
df['Country'] = df['Country'].replace('', np.nan)
df['Country'] = df['Country'].fillna('Unknown')


# In[28]:


df


# In[29]:


# 8. Clean Feedback (fill blanks or NaNs with 'No Feedback')
df['Feedback'] = df['Feedback'].replace('', np.nan)
df['Feedback'] = df['Feedback'].fillna('No Feedback')


# In[30]:


df


# In[31]:


# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

