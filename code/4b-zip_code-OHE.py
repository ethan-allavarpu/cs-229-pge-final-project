#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Read in training data
train_data = pd.read_csv(
    '../data/processed/x_train.csv', index_col=0, dtype=str
)
train_data.info()


# In[4]:


# Create one-hot encoded categorical variables for zip codes and add columns to dataframe
zips_OHE = pd.get_dummies(train_data.zip_code, prefix='zip_is')
train_w_OHE = train_data.merge(
  zips_OHE, how = 'inner', left_index=True, right_index=True)


# In[5]:


# Read in test data file
test_data_w_OHE = pd.read_csv(
    '../data/processed/x_test.csv', index_col=0, dtype=str)
print(test_data_w_OHE.columns.size)
print(train_w_OHE.columns.size)

# Populate test dataset with OHE zip code columns from training set 
for i in train_w_OHE.columns:
    if i not in test_data_w_OHE.columns:
        new_col = np.zeros(len(test_data_w_OHE))
        for j in range(len(test_data_w_OHE)):
            # Populate a 1 in the new OHE zip code column for examples from that zip code
            if test_data_w_OHE.iloc[j]['zip_code'] == i[7:]: 
                new_col[j] = 1
        test_data_w_OHE[i] = new_col[:]
print(test_data_w_OHE.columns.size)
print(train_w_OHE.columns.size)


# In[6]:


# Percent of test observations without ZIP code in training set columns
print("Percent of test observations without ZIP code present in training set")
print(np.mean(np.array(test_data_w_OHE.iloc[:, -254:]).sum(axis = 1) == 0))


# In[7]:


# Write OHE training and test set to CSV
train_w_OHE.to_csv("../data/processed/x_train_w_OHE.csv")
test_data_w_OHE.to_csv("../data/processed/x_test_w_OHE.csv")

