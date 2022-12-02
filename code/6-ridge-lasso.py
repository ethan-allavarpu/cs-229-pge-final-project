#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


# In[2]:


x_train = pd.read_csv(
  "../data/processed/x_train_w_OHE.csv", index_col=0, dtype=str
).reset_index(drop=True)
x_test = pd.read_csv(
  "../data/processed/x_test_w_OHE.csv", index_col=0, dtype=str
).reset_index(drop=True)
y_train = pd.read_csv(
  "../data/processed/y_train.csv", index_col=0, dtype=float
).squeeze("columns")
y_test = pd.read_csv(
  "../data/processed/y_test.csv", index_col=0, dtype=float
).squeeze("columns")


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


scaler = StandardScaler()
scaler.fit(rel_x_train)
scaled_train_x = scaler.transform(rel_x_train)
scaled_test_x = scaler.transform(rel_x_test)


# In[5]:


# Range of penalties for ridge
alphas = 10 ** np.arange(-7., 2.)


# In[6]:


ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(scaled_train_x, y_train)
ridge_preds = ridge.predict(scaled_test_x)


# In[7]:


lasso = LassoCV(max_iter=int(1e6), cv=5, random_state=6)
lasso.fit(scaled_train_x, y_train)
lasso_preds = lasso.predict(scaled_test_x)


# In[8]:


# Elastic net is like a hybrid between LASSO and ridge
e_net = ElasticNetCV(
    l1_ratio=[0.01, .1, .3, .5, .65, .8, .9, .95, .975, .99, 1],
    max_iter=int(1e6), cv=5, random_state=6
)
e_net.fit(scaled_train_x, y_train)
e_net_preds = e_net.predict(scaled_test_x)


# In[9]:


def calc_test_r2(pred_vals, true_vals, baseline_rmse):
    sse = mean_squared_error(pred_vals, true_vals) * len(true_vals)
    sst = (baseline_rmse ** 2) * len(true_vals)
    return (
        1 - sse / sst, np.sqrt(sse / len(true_vals)),
        mean_absolute_error(pred_vals, true_vals)
    )


# In[10]:


# See model performance for all three regularization models
baseline_rmse = np.sqrt(((y_test - y_test.mean()) ** 2).mean())
regularization_results = pd.DataFrame({
    'model': ['Ridge', 'LASSO', 'Elastic Net'],
    'stats': [
        calc_test_r2(preds, y_test, baseline_rmse)
        for preds in [ridge_preds, lasso_preds, e_net_preds]
    ]
})
regularization_results['test_r_sq'] = [
    model[0] for model in regularization_results['stats']
]
regularization_results['rmse'] = [
    model[1] for model in regularization_results['stats']
]
regularization_results['mae'] = [
    model[2] for model in regularization_results['stats']
]
regularization_results.drop(columns='stats', inplace=True)
print(regularization_results)

