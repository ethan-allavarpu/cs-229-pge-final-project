#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


# Read in training and test datasets
x_train = pd.read_csv(
  "../data/processed/x_train_w_OHE.csv", index_col=0, dtype=str)
x_test = pd.read_csv(
  "../data/processed/x_test_w_OHE.csv", index_col=0, dtype=str)
y_train = pd.read_csv(
  "../data/processed/y_train.csv", index_col=0, dtype=float
  ).squeeze("columns").reset_index(drop=True)
y_test = pd.read_csv(
  "../data/processed/y_test.csv", index_col=0, dtype=float
).squeeze("columns").reset_index(drop=True)


# In[3]:


# Ensure features are of correct type for modeling
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
zip_cols = x_train.columns[
    [re.search('zip_is', col) is not None for col in x_train.columns]
]
x_train = get_correct_types_x(x_train, numeric_cols)
x_test = get_correct_types_x(x_test, numeric_cols)
rel_x_train = x_train[numeric_cols]
rel_x_test = x_test[numeric_cols]


# In[4]:


# Scale all numeric columns with same transformation as training set
scaler = StandardScaler()
scaler.fit(rel_x_train)
scaled_train_x = scaler.transform(rel_x_train)
scaled_test_x = scaler.transform(rel_x_test)


# In[5]:


# Read in predictions from final xgboost model
best_preds = np.loadtxt("../data/predictions/xgboost_final_preds.csv")


# In[6]:


# Calculate residuals
resids = y_test - best_preds


# In[7]:


# Take 10% of observations with worst predictions
top10_pct_off = np.argsort(np.abs(np.array(resids)))[-int(0.1 * len(resids))::][::-1]
extreme_resids = resids[top10_pct_off]
extreme_x = x_test.iloc[top10_pct_off, :]
extreme_y = y_test[top10_pct_off]

# Take remaining 90% predictions for feature comparison
other_90_idx = np.argsort(np.abs(np.array(resids)))[ : int(0.9 * len(resids))]
other_90_x = x_test.iloc[other_90_idx, :]
other_90_y = y_test[other_90_idx]


# In[8]:


# Percentage of zip codes resulting in "extreme" errors vs. percentage 
# appearing in the full data set
zip_code_errors = pd.merge(
  (extreme_x.zip_code.value_counts() /
    extreme_x.zip_code.value_counts().sum()),
  (x_test.zip_code.value_counts() / 
    x_test.zip_code.value_counts().sum()),
  how='inner', left_index=True, right_index=True, 
).rename(columns={'zip_code_x': 'extreme', 'zip_code_y': 'all'})
zip_code_errors['pct_diff'] = zip_code_errors['extreme']/ zip_code_errors['all']
# zip_code_errors['total_pop'] =  # ADD TOTAL_POP FROM X_TEST TO THIS DATAFRAME?

print(zip_code_errors.sort_values('pct_diff', ascending=False).iloc[0:25,])


# In[ ]:


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


# In[9]:


# Plot histogram of prediction errors
sns.histplot(resids)
plt.title("Histogram of Errors, XGBoost", size=12)
plt.xlabel("Actual - Predicted Outage", size=10)
plt.savefig("../visuals/xgb_errors.png")
plt.show()


# In[14]:


# View dataframe of observations resulting in 10% worst predictions 
extreme_x.head(15)


# In[11]:


# Compare differences in numeric features among observations with 10% 
# largest error versus those with 90% smallest errors
numeric_diffs = pd.DataFrame(
  [(feat, np.median(extreme_x[feat]), np.median(other_90_x[feat]))
   for feat in numeric_cols],
  columns=['feature', 'extreme_median', 'other_90_median']
)
# Calculate differences as percentage and sort by abs val of largest % difference
numeric_diffs['pct_diff'] = (
  (numeric_diffs.extreme_median - numeric_diffs.other_90_median) / 
   numeric_diffs.other_90_median)
numeric_diffs['abs_diff'] = np.abs(numeric_diffs.pct_diff)
print(
  numeric_diffs.sort_values('abs_diff', ascending=False). \
    drop(columns=['abs_diff']).reset_index(drop=True)
)

