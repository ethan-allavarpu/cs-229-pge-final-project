#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


# In[2]:


all_shutoffs = pd.read_csv(
    '../data/processed/temp-shutoffs.csv', index_col=0, dtype=str
)
nonmissing = all_shutoffs[~all_shutoffs.zip_code.isnull()]
nonmissing.deenergize_time = nonmissing.deenergize_time.\
    apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
nonmissing.restoration_time = nonmissing.restoration_time.\
    apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
updated_missing = pd.read_csv(
    '../data/processed/updated-missing-zips.csv', index_col=0, dtype=str
).set_index('index')
# Convert latitude/longitude to floats (+/-)
updated_missing.longitude = updated_missing.longitude.\
    apply(lambda x: -float(re.sub('W', '', x)))
updated_missing.latitude = updated_missing.latitude.\
    apply(lambda x: float(re.sub('N', '', x)))

# Convert to datetime
updated_missing.deenergize_time = updated_missing.deenergize_time.\
    apply(lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))
updated_missing.restoration_time = updated_missing.restoration_time.\
    apply(lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M'))


# In[3]:


psps_shutoffs = pd.concat([nonmissing, updated_missing]).reset_index(drop=True)
psps_shutoffs['substn_present'] = psps_shutoffs['Substation Name'] != 'MISSING'
psps_shutoffs.drop(columns='Substation Name', inplace=True)


# In[4]:


for col in ['time_out_min','hftd_tier', 'total_affected',  
            'residential_affected', 'longitude', 'latitude']:
    psps_shutoffs[col] = psps_shutoffs[col].astype(float)


# In[5]:


census_data = pd.read_csv('../data/raw/2020-population.csv')
census_data = census_data[
    [column for column in census_data.columns if re.search('E$', column)]
]
census_pop = census_data[['NAME', 'DP05_0001E', 'DP05_0018E', 'DP05_0037E']] # ZCTA, total population cols, median age, white population
census_pop.columns = ['name', 'total_pop', 'median_age', 'white_pop']
census_pop.drop(index=0, inplace=True)
census_pop['ZCTA'] = [re.findall('\d{5}', obs)[0] for obs in census_pop.name]

income_data = pd.read_csv('../data/raw/2020-median-income.csv')
census_income = income_data[['NAME', 'S1901_C01_012E']] # ZCTA, median HH income
census_income.columns = ['name', 'median_income']
census_income.drop(index=0, inplace=True)
census_income['ZCTA'] = [re.findall('\d{5}', obs)[0] for obs in census_income.name]

census_pop_income = pd.merge(census_pop, census_income, how='inner', on='ZCTA')

# Read in converter between ZCTA (census) and ZIP
zip_zcta = pd.read_excel(
    '../data/raw/zip-code-zcta.xlsx', dtype='str'
)[['ZIP_CODE', 'ZCTA']]

def convert_to_float(x):
    if type(x) == str:
        return re.sub('[+,-]*', '', x) if x != '-' else 'nan'
    return x
# Join the two data sets
zip_census = pd.merge(census_pop_income, zip_zcta, how='inner', on='ZCTA')
for col in ['total_pop', 'median_age', 'white_pop', 'median_income']:
    zip_census[col] = zip_census[col].apply(convert_to_float).astype(float)
zip_census['white_pct'] = zip_census.white_pop / zip_census.total_pop


# In[7]:


# Merge shutoff data with zip code and census information
shutoffs_pop = pd.merge(
    psps_shutoffs, zip_census[
        ['ZIP_CODE', 'total_pop', 'median_age', 'median_income', 'white_pct']
    ],
    how='left', left_on='zip_code', right_on='ZIP_CODE'
).drop(columns='ZIP_CODE')
shutoffs_pop.to_csv('../data/processed/processed-shutoffs.csv', index=False)

