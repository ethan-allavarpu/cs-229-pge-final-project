#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd


# In[2]:


processed_shutoffs = pd.read_csv(
    '../data/processed/processed-shutoffs.csv', dtype=str)
processed_shutoffs['deenergize_time_date'] = pd.to_datetime(processed_shutoffs['deenergize_time']).dt.date

weather = pd.DataFrame(columns = ['index', 'circuit_name', 'deenergize_time_date', 'tmin_d-5', 'tmax_d-5', 'wspd_d-5',
'tmin_d-4', 'tmax_d-4', 'wspd_d-4', 'tmin_d-3', 'tmax_d-3', 'wspd_d-3',
'tmin_d-2', 'tmax_d-2', 'wspd_d-2', 'tmin_d-1', 'tmax_d-1', 'wspd_d-1'])

raw_weather_files = os.listdir('../data/weather/raw/daily/')
# Add the weather data from 5 days before to the day before
for i in raw_weather_files:
    deenergize_time = pd.to_datetime(i.split('_')[3][0:-4])
    new_circuit_row = pd.DataFrame(columns = ['index', 'circuit_name', 'deenergize_time_date', 'tmin_d-5', 'tmax_d-5', 'wspd_d-5',
    'tmin_d-4', 'tmax_d-4', 'wspd_d-4', 'tmin_d-3', 'tmax_d-3', 'wspd_d-3',
    'tmin_d-2', 'tmax_d-2', 'wspd_d-2', 'tmin_d-1', 'tmax_d-1', 'wspd_d-1'], index=range(1))
    new_circuit_row['index'] = int(i.split('_')[1])
    new_circuit_row['circuit_name'] = i.split('_')[2] 
    new_circuit_row['deenergize_time_date'] = deenergize_time
    temp_df = pd.read_csv('../data/weather/raw/daily/' + i)
    temp_df['time'] = pd.to_datetime(temp_df['time'])
    for j in range(len(temp_df)):
        # print(temp_df.iloc[j]['time'])
        if (deenergize_time - temp_df.iloc[j]['time']).days == 5:
            new_circuit_row['tmin_d-5'] = temp_df.iloc[j]['tmin']
            new_circuit_row['tmax_d-5'] = temp_df.iloc[j]['tmax']
            new_circuit_row['wspd_d-5'] = temp_df.iloc[j]['wspd']
            # new_circuit_row.extend([temp_df.iloc[j]['tmin'], temp_df.iloc[j]['tmax'], temp_df.iloc[j]['wspd']])
        elif (deenergize_time - temp_df.iloc[j]['time']).days == 4:
            new_circuit_row['tmin_d-4'] = temp_df.iloc[j]['tmin']
            new_circuit_row['tmax_d-4'] = temp_df.iloc[j]['tmax']
            new_circuit_row['wspd_d-4'] = temp_df.iloc[j]['wspd']
            # new_circuit_row.extend([temp_df.iloc[j]['tmin'], temp_df.iloc[j]['tmax'], temp_df.iloc[j]['wspd']])
        elif (deenergize_time - temp_df.iloc[j]['time']).days == 3:
            new_circuit_row['tmin_d-3'] = temp_df.iloc[j]['tmin']
            new_circuit_row['tmax_d-3'] = temp_df.iloc[j]['tmax']
            new_circuit_row['wspd_d-3'] = temp_df.iloc[j]['wspd']
            # new_circuit_row.extend([temp_df.iloc[j]['tmin'], temp_df.iloc[j]['tmax'], temp_df.iloc[j]['wspd']])
        elif (deenergize_time - temp_df.iloc[j]['time']).days == 2:
            new_circuit_row['tmin_d-2'] = temp_df.iloc[j]['tmin']
            new_circuit_row['tmax_d-2'] = temp_df.iloc[j]['tmax']
            new_circuit_row['wspd_d-2'] = temp_df.iloc[j]['wspd']
            # print(temp_df.iloc[j]['time'].date, 'deenergize-1', deenergize_time.date)
            # new_circuit_row.extend([temp_df.iloc[j]['tmin'], temp_df.iloc[j]['tmax'], temp_df.iloc[j]['wspd']])
        elif (deenergize_time - temp_df.iloc[j]['time']).days == 1:
            new_circuit_row['tmin_d-1'] = temp_df.iloc[j]['tmin']
            new_circuit_row['tmax_d-1'] = temp_df.iloc[j]['tmax']
            new_circuit_row['wspd_d-1'] = temp_df.iloc[j]['wspd']
            # new_circuit_row.extend([temp_df.iloc[j]['tmin'], temp_df.iloc[j]['tmax'], temp_df.iloc[j]['wspd']])
    weather = pd.concat([weather, new_circuit_row], ignore_index=True)
weather.set_index('index', inplace=True)


# In[3]:


processed_shutoffs_weather = processed_shutoffs.merge(weather, how = 'inner', left_index=True, right_index=True)

print("length of dataframe: ", len(processed_shutoffs_weather))
processed_shutoffs_weather.info()


# In[4]:


for col in ['deenergize_time', 'restoration_time']:
    processed_shutoffs_weather[col] = pd.to_datetime(processed_shutoffs_weather[col], format='%Y-%m-%d %H:%M:%S')
print(processed_shutoffs_weather.columns)
# Verify matching
print(all(processed_shutoffs_weather['circuit_name_x'] == processed_shutoffs_weather['circuit_name_y']))
print(all(processed_shutoffs_weather['deenergize_time_date_x'] == processed_shutoffs_weather['deenergize_time_date_y']))


# In[5]:


# Drop duplicate columns
processed_shutoffs_weather.drop(columns=['deenergize_time_date_y', 'deenergize_time_date_x', 'circuit_name_y'], inplace=True)
processed_shutoffs_weather.rename(columns={'circuit_name_x':'circuit_name'}, inplace=True)


# In[6]:


processed_shutoffs_weather.to_csv('../data/processed/processed-shutoffs-weather.csv', index=False)

