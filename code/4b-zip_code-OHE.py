#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_data = pd.read_csv(
    '../data/processed/x_train.csv', index_col=0, dtype=str
)


# In[3]:


zips_OHE = pd.get_dummies(train_data.zip_code, prefix='zip_is')
train_w_OHE = train_data.merge(
  zips_OHE, how = 'inner', left_index=True, right_index=True
)


# In[4]:


test_data_w_OHE = pd.read_csv(
    '../data/processed/x_test.csv', index_col=0, dtype=str
)
print(test_data_w_OHE.columns.size)
print(train_w_OHE.columns.size)
# See if we have a new zip code
for i in train_w_OHE.columns:
    if i not in test_data_w_OHE.columns:
        new_col = np.zeros(len(test_data_w_OHE))
        for j in range(len(test_data_w_OHE)):
            if test_data_w_OHE.iloc[j]['zip_code'] == i[7:]:
                new_col[j] = 1
        test_data_w_OHE[i] = new_col[:]
print(test_data_w_OHE.columns.size)
print(train_w_OHE.columns.size)


# In[5]:


# Percent of test observations without ZIP code in training
print("Percent of test observations without ZIP code present in training set")
print(np.mean(np.array(test_data_w_OHE.iloc[:, -254:]).sum(axis = 1) == 0))


# In[6]:


train_w_OHE.to_csv("../data/processed/x_train_w_OHE.csv")
test_data_w_OHE.to_csv("../data/processed/x_test_w_OHE.csv")

