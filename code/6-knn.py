#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# In[2]:


x_train = pd.read_csv(
  "../data/processed/x_train.csv", index_col=0, dtype=str
)
x_test = pd.read_csv(
  "../data/processed/x_test.csv", index_col=0, dtype=str
)
y_train = pd.read_csv(
  "../data/processed/y_train.csv", index_col=0, dtype=float
).squeeze("columns").reset_index(drop=True)
y_test = pd.read_csv(
  "../data/processed/y_test.csv", index_col=0, dtype=float
).squeeze("columns").reset_index(drop=True)


# In[3]:


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


# In[4]:


# Scale to zero mean, variance 1
scaler = StandardScaler()
scaler.fit(rel_x_train)
scaled_train_x = scaler.transform(rel_x_train)
scaled_test_x = scaler.transform(rel_x_test)


# In[5]:


k_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 40, 50]
k_scores = np.zeros((len(k_vals), 5))


# In[6]:


kf = KFold(n_splits=5, shuffle=True, random_state=229) # 5-fold CV
cv_iter = 0
# K-fold cross validation for best k value for training set
for train, test in kf.split(scaled_train_x):
    cv_train_x = scaled_train_x[train]
    cv_train_y = y_train[train]
    cv_test_x = scaled_train_x[test]
    cv_test_y = y_train[test]
    k_iter = 0
    for k in k_vals:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(cv_train_x, cv_train_y)
        preds = knn.predict(cv_test_x)
        k_scores[k_iter, cv_iter] = mean_squared_error(cv_test_y, preds)
        k_iter += 1
    cv_iter += 1
cv_scores = k_scores.mean(axis=1)


# In[7]:


print('Best k:', k_vals[np.argmin(cv_scores)])


# In[8]:


# Use optimal K for final model
best_knn = KNeighborsRegressor(n_neighbors=k_vals[np.argmin(cv_scores)])
best_knn.fit(scaled_train_x, y_train)
knn_preds = best_knn.predict(scaled_test_x)


# In[9]:


def calc_test_r2(pred_vals, true_vals, baseline_rmse):
    sse = mean_squared_error(pred_vals, true_vals) * len(true_vals)
    sst = (baseline_rmse ** 2) * len(true_vals)
    return (
        1 - sse / sst, np.sqrt(sse / len(true_vals)),
        mean_absolute_error(pred_vals, true_vals)
    )


# In[10]:


baseline_rmse = np.sqrt(((y_test - y_test.mean()) ** 2).mean())
test_r2, rmse, mae = calc_test_r2(knn_preds, y_test, baseline_rmse)
print('Test R-Squared:', test_r2)
print('RMSE:', rmse)
print('MAE:', mae)

