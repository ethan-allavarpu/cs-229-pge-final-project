#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[8]:


psps = pd.read_csv(
    '../data/processed/processed-shutoffs-weather.csv', dtype=str
).dropna()
# Proper datetime conversion
for col in ['deenergize_time', 'restoration_time']:
    psps[col] = pd.to_datetime(psps[col], format='%Y-%m-%d %H:%M:%S')
numeric_cols = [
    'time_out_min', 'hftd_tier', 'total_affected', 'residential_affected',
    'longitude', 'latitude', 'total_pop', 'median_age', 'median_income',
    'white_pct', 'tmin_d-5', 'tmax_d-5', 'wspd_d-5', 'tmin_d-4', 'tmax_d-4',
    'wspd_d-4', 'tmin_d-3', 'tmax_d-3', 'wspd_d-3', 'tmin_d-2', 'tmax_d-2',
    'wspd_d-2', 'tmin_d-1', 'tmax_d-1', 'wspd_d-1'
]
# Proper float conversion
for col in numeric_cols:
    psps[col] = psps[col].astype(float)
# Add day in year column (for the possibility of cyclical patterns/fire season)
psps['day_in_year'] = [int(day) for day in psps.deenergize_time.dt.day_of_year]


# In[3]:


# Constant train-test split for reproducibility
# 80-20 train-test
x_train, x_test, y_train, y_test = train_test_split(
    psps.drop(columns='time_out_min'),
    psps.time_out_min,
    test_size=0.2, random_state=229
)


# In[4]:


x_train.to_csv("../data/processed/x_train.csv")
x_test.to_csv("../data/processed/x_test.csv")
y_train.to_csv("../data/processed/y_train.csv")
y_test.to_csv("../data/processed/y_test.csv")


# In[6]:


# Make sure preliminary models test on the same features as they did for the milestone
drop_cols = ['day_in_year', 'median_age', 'median_income', 'white_pct',
             'tmin_d-5', 'tmax_d-5', 'wspd_d-5']
x_train.drop(columns=drop_cols).to_csv("../data/processed/x_train-prelim.csv")
x_test.drop(columns=drop_cols).to_csv("../data/processed/x_test-prelim.csv")

