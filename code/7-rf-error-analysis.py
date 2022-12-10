#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[3]:


x_train = pd.read_csv(
  "../data/processed/x_train.csv", index_col=0, dtype=str
).reset_index(drop=True)
x_test = pd.read_csv(
  "../data/processed/x_test.csv", index_col=0, dtype=str
).reset_index(drop=True)
y_train = pd.read_csv(
  "../data/processed/y_train.csv", index_col=0, dtype=float
).squeeze("columns").reset_index(drop=True)
y_test = pd.read_csv(
  "../data/processed/y_test.csv", index_col=0, dtype=float
).squeeze("columns").reset_index(drop=True)


# In[4]:


# Proper type conversion
def get_correct_types_x(df, numeric_cols):
    for col in ['deenergize_time', 'restoration_time']:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    return df
numeric_cols = [
    'hftd_tier', 'total_affected', 'residential_affected',
    'longitude', 'latitude', 'total_pop', 'median_age', 'median_income',
    'white_pct', 'tmin_d-5', 'tmax_d-5', 'wspd_d-5', 'tmin_d-4', 'tmax_d-4',
    'wspd_d-4', 'tmin_d-3', 'tmax_d-3', 'wspd_d-3', 'tmin_d-2', 'tmax_d-2',
    'wspd_d-2', 'tmin_d-1', 'tmax_d-1', 'wspd_d-1', 'day_in_year'
]
x_train = get_correct_types_x(x_train, numeric_cols)
x_test = get_correct_types_x(x_test, numeric_cols)
rel_x_train = x_train[numeric_cols]
rel_x_test = x_test[numeric_cols]


# In[5]:


# Zero mean, variance 1
scaler = StandardScaler()
scaler.fit(rel_x_train)
scaled_train_x = scaler.transform(rel_x_train)
scaled_test_x = scaler.transform(rel_x_test)


# In[6]:


# Read in RF predictions and calculate residuals as observed - predicted
best_preds = np.loadtxt("../data/predictions/rf_preds.csv")
resids = y_test - best_preds


# In[7]:


# Take 10% most incorrect predictions
top10_pct_off = np.argsort(
  np.abs(np.array(resids))
)[-int(0.1 * len(resids))::][::-1]
extreme_resids = resids[top10_pct_off]
extreme_x = x_test.iloc[top10_pct_off, :]
extreme_y = y_test[top10_pct_off]


# In[8]:


# Fraction of "extreme" errors from a given date vs. fraction for all
print('Fraction of observations in ')
print(
  pd.merge(
    (
      extreme_x.deenergize_time.dt.date.value_counts() /
      extreme_x.deenergize_time.dt.date.value_counts().sum()
    ),
    (
      x_test.deenergize_time.dt.date.value_counts() / 
      x_test.deenergize_time.dt.date.value_counts().sum()
    ),
    how='inner', left_index=True, right_index=True
  ).rename(
    columns={'deenergize_time_x': 'extreme', 'deenergize_time_y': 'all'}
  ).sort_values('extreme', ascending=False)
)


# In[13]:


sns.histplot(resids)
plt.title("Histogram of Errors, Random Forest")
plt.xlabel("Actual - Predicted Outage")
plt.savefig("../data/predictions/rf_errors.png")
plt.show()


# In[10]:


# Extremely incorrect point (residual around 6K)
print("Most extreme residual")
print(x_test.iloc[np.argmax(np.abs(resids)), :])


# In[11]:


def compare_extreme_vals(all, idx, function):
  """
  Compare the extreme values (by index) against all others for a given function
  """
  flip_idx = np.ones(len(all), dtype=bool)
  flip_idx[idx] = False
  extreme = all[idx]
  others = all[flip_idx]
  if all.dtype == float:
    return np.round(function(extreme), 4), np.round(function(others), 4)
  elif all.dtype == '<M8[ns]':
    combined_counts = pd.merge(
      (extreme.dt.date.value_counts() / extreme.dt.date.value_counts().sum()),
      (others.dt.date.value_counts() / others.dt.date.value_counts().sum()),
      how='inner', left_index=True, right_index=True
    )
    combined_counts.columns = ['extreme', 'others']
    return combined_counts.sort_values('extreme', ascending=False)
  elif all.dtype == 'O':
    combined_counts = pd.merge(
      (extreme.value_counts() / extreme.value_counts().sum()),
      (others.value_counts() / others.value_counts().sum()),
      how='inner', left_index=True, right_index=True
    )
    combined_counts.columns = ['extreme', 'others']
    return combined_counts.sort_values('extreme', ascending=False)


# In[12]:


# Print Numeric Features
numeric_diffs = pd.DataFrame(
  [(feat, compare_extreme_vals(x_test[feat], top10_pct_off, np.median))
   for feat in x_test.columns if x_test[feat].dtype == float],
  columns=['feature', 'medians']
)
# Extract medians for extreme vs less extreme residuals
numeric_diffs['extreme'] = [feat[0] for feat in numeric_diffs.medians]
numeric_diffs['others'] = [feat[1] for feat in numeric_diffs.medians]
numeric_diffs['pct_diff'] = (
  (numeric_diffs.extreme - numeric_diffs.others) / numeric_diffs.others
)
numeric_diffs['abs_diff'] = np.abs(numeric_diffs.pct_diff)
# Get percent differences to see patterns in incorrect responses
print('\nDifference in medians between extreme and other residuals')
print('-----------------------------------------------------------')
print(
  numeric_diffs.sort_values('abs_diff', ascending=False).\
    drop(columns=['medians', 'abs_diff']).reset_index(drop=True)
)

