#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
from fuzzywuzzy import fuzz
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')


# In[2]:

# Standardize substation names
substations = pd.read_csv('../data/raw/California_Electric_Substations.csv',
                          usecols=['Name', 'ZIP_CODE', 'COUNTY', 'Lon', 'Lat'], dtype='str')
substations.Name = [stn.upper() for stn in substations.Name]
substations = substations[substations.Name != 'UNKNOWN'].reset_index(drop=True)
substations = substations[
    [re.match('^\d', name) is None for name in substations.Name]
]
substations


# In[3]:


def clean_df(psps, start_datetype, end_datetype):
    # Standardize column names
    if 'Distribution / Transmission' in psps.columns:
        psps['Distribution / Transmission'] = [
            x.upper() for x in psps['Distribution / Transmission']
        ]
        psps = psps[psps['Distribution / Transmission'] == 'DISTRIBUTION']
        psps.drop(columns='Distribution / Transmission', inplace=True)
    # Clean circuit names
    cleaned_circuit = [
        re.findall('.*?(?=\s\d{4,}\*?)', str(circuit))[0].strip()
        if len(re.findall('.*?(?=\s\d{4,}\*?)', str(circuit))) > 0 else circuit
        for circuit in psps['Circuit Name']
    ]
    psps['Circuit Name'] = cleaned_circuit
    # Clean fire risk tiers
    cleaned_hftd = [
        int(max(re.findall('\d', str(hftd))))
        if len(re.findall('\d', str(hftd))) > 0 else 0
        for hftd in psps['HFTD Tier(s)']
    ]
    psps['HFTD Tier'] = cleaned_hftd
    psps.columns = [str(col_name).strip() for col_name in psps.columns]
    shorter_names = [
        re.sub(' Customers$', '', str(col_name)) for col_name in psps.columns
    ]
    psps.columns = shorter_names
    # Standardize column names
    psps.rename(columns={
        'Start Date and\rTime': 'DeEnergization Date and Time',
        'Start Date and Time': 'DeEnergization Date and Time',
        'De-Energization Date and Time (PDT)': 'DeEnergization Date and Time',
        'De-Energization Date and Time': 'DeEnergization Date and Time',
        'Restoration Date and Time (PDT)': 'Restoration Date and Time',
        'Commercial/Industrial': 'Commercial / Industrial'
        },
        inplace=True
    )
    if 'Key Communities' not in psps.columns:
        psps.rename(columns={'Counties': 'Key Communities'}, inplace=True)
    fixed_communities = [
        re.sub('[\r\n\s]+', ' ', str(comm)) for comm in psps['Key Communities']
    ]
    # Get the datetime-variables
    def get_times(str_time, date_type):
        fixed_time = re.sub('[\r\n\s]+', ' ', str(str_time))
        try:
            time = pd.to_datetime(fixed_time, format=date_type)
        except:
            try:
                date_format = re.sub('y', 'Y', date_type)
                time = pd.to_datetime(fixed_time, format=date_format)
            except:
                if re.search('\d', fixed_time) is None:
                    time = pd.to_datetime('2000-01-01')
                else:
                    time = pd.to_datetime('1970-01-01')
        return time
    start_time = [
        get_times(time, start_datetype)
        for time in psps['DeEnergization Date and Time']
    ]
    end_time = [
        get_times(time, end_datetype)
        for time in psps['Restoration Date and Time']
    ]
    psps['deenergize_time'] = start_time
    psps['restoration_time'] = end_time
    psps['time_out_min'] = (
        (psps.restoration_time - psps.deenergize_time) / pd.Timedelta('1m')
    )
    psps['Key Communities'] = fixed_communities
    for col in ['HFTD Tier', 'Total', 'Residential']:
        psps[col] = [float(re.sub(',', '', str(val))) for val in psps[col]]
    psps = psps[[    
        'Circuit Name', 'deenergize_time', 'restoration_time', 'time_out_min',
        'Key Communities', 'HFTD Tier', 'Total', 'Residential'
    ]]
    psps.columns = [
        'circuit_name', 'deenergize_time', 'restoration_time', 'time_out_min', 'key_communities', 'hftd_tier', 'total_affected', 'residential_affected'
    ]
    # Fill in potential missing values
    def fill_values(missing_obs, non_missing_data):
        temp_non_missing = copy.deepcopy(non_missing_data)
        temp_non_missing['cleaned_circuit'] = [
            re.sub('-?\d+$', '', circuit).strip()
            for circuit in temp_non_missing.circuit_name
        ]
        missing_circuit = re.sub('-?\d+$', '', missing_obs.circuit_name).strip()
        rel_circ = temp_non_missing[
            temp_non_missing.cleaned_circuit == missing_circuit
        ]
        if len(rel_circ) > 0:
            missing_obs.deenergize_time = rel_circ.deenergize_time.mean()
            missing_obs.restoration_time = rel_circ.restoration_time.mean()
            missing_obs.time_out_min = round(rel_circ.time_out_min.mean())
            missing_obs.key_communities = ', '.join(rel_circ.key_communities)
        return missing_obs
    missing = psps[psps.deenergize_time.isna()]
    if len(missing) > 0:
        non_missing = psps[psps.deenergize_time > pd.to_datetime('2000-01-01')]
        psps = pd.concat(
            [non_missing,
            pd.DataFrame([fill_values(row, non_missing)
                          for _, row in missing.iterrows()])],
            axis=0
        )
    return psps


# In[4]:

# Treat yearly data differently
# Each year has a slightly different format 
def clean_df_20_21(file_path, start_datetype, end_datetype):
    psps = pd.read_csv(file_path, dtype=str)
    return clean_df(psps, start_datetype, end_datetype)

# Have to merge files for 2018-2019 before cleaning them
def clean_df_18_19(psps_date, start_datetype, end_datetype):
    circuits = pd.read_csv(
        '../data/raw/PSPS-{}-circuits.csv'.format(psps_date), dtype=str
    ).rename(columns={'Circuit': 'Circuit Name'})
    circuits['Circuit Name'] = [
        re.sub('\s(?=\d)', '', re.sub('\*', '', re.sub('[\r\n]+', '', circuit)))
        for circuit in circuits['Circuit Name']
    ]
    customers = pd.read_csv(
        '../data/raw/PSPS-{}-customers.csv'.format(psps_date), dtype=str
    ).rename(
        columns={'Impacted Circuit': 'Circuit Name', 'Circuit': 'Circuit Name'}
    )
    customers['Circuit Name'] = [
        re.sub('\s(?=\d)', '', re.sub('\*', '', re.sub('[\r\n]+', '', circuit)))
        for circuit in customers['Circuit Name']
    ]
    psps = pd.merge(circuits, customers, how='outer', on='Circuit Name')
    psps = psps[[
        re.search('LINE$', circuit) is None
        for circuit in psps['Circuit Name']
    ]]
    return clean_df(psps, start_datetype, end_datetype)


# In[5]:

# Read in 2018-2019 files
dates_18_19 = [
    '10.14.18', '06.08.19', '09.23.19', '10.05.19',
    '10.09.19', '10.23.19', '10.26.19', '11.20.19'
]
start_formats = [
    '%m/%d/%y %H:%M', '%m/%d/%y %H:%M', '%m/%d/%y %H:%M', '%m/%d/%y %H:%M',
    '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M'
]
end_formats = [
    '%m/%d/%y %H:%M', '%m/%d/%y %H:%M', '%m/%d/%y %H:%M', '%m/%d/%y %H:%M',
    '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M'
]

data_input = [
    (dates_18_19[i], start_formats[i], end_formats[i])
    for i in range(len(dates_18_19))
]

data_18_19 = pd.concat(
    [clean_df_18_19(file[0], file[1], file[2]) for file in data_input],
    axis=0
).reset_index(drop=True)


# In[6]:


file_names = [
    '../data/raw/PSPS-{}-circuits.csv'.format(dt)
    for dt in [
        '09.07.20', '09.27.20', '10.14.20', '10.21.20', '10.25.20', '12.02.20',
        '01.19.21', '08.17.21', '09.20.21', '10.11.21', '10.14.21'
    ]
]
start_formats = [
    '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M', '%d/%m/%y %H:%M',
    '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%d/%m/%y %H:%M',
    '%d/%m/%y %H:%M', '%d/%m/%y %H:%M', '%d/%m/%y %H:%M'
]
end_formats = [
    '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M', '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S',
    '%m/%d/%Y %H:%M', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%d/%m/%y %H:%M',
    '%d/%m/%y %H:%M', '%d/%m/%y %H:%M', '%d/%m/%y %H:%M'
]
data_input = [(file_names[i], start_formats[i], end_formats[i]) for i in range(len(file_names))]

# Read in 2020-2021 files
data_20_21 = pd.concat(
    [clean_df_20_21(file[0], file[1], file[2]) for file in data_input],
    axis=0
).reset_index(drop=True)


# In[7]:

# Subset to valid results
data = pd.concat([data_18_19, data_20_21], axis=0)
data = data[
        (data.restoration_time > '2000-01-01') &
        (data.deenergize_time > '2000-01-01') &
        (~data.total_affected.isnull()) &
        (~data.total_affected.isna())
    ].reset_index(drop=True)


# In[8]:

# Use fuzzymatching to assign the "correct" substation (for joining to weather)
def most_similar_station(circuit, stns, thresh=80):
    circuit = re.sub('\.', '', circuit)
    circuit = re.sub(' NO', '', circuit)
    circuit = re.sub('\d', '', circuit).strip()
    sim_scores =[
        (
            fuzz.token_sort_ratio(
                circuit,
                re.sub(' TAP', '', re.sub('[\d&]', '', stn.Name)).strip()
            ),
            stn.Name, stn.ZIP_CODE, stn.Lon, stn.Lat
        )
        for _, stn in stns.iterrows()
    ]
    max_score = max([scores[0] for scores in sim_scores])
    if max_score < thresh:
        return ('Default', None, None, None)
    return [scores[1:] for scores in sim_scores if scores[0] == max_score][0]


# In[9]:


unique_circuits = data['circuit_name'].unique()
closest_substation = {
    circuit: most_similar_station(circuit, substations)
    for circuit in unique_circuits
}


# In[10]:

# Unpack zip, latitude, longitude
zips = [closest_substation[circuit][1] for circuit in data['circuit_name']]
longs = [closest_substation[circuit][2] for circuit in data['circuit_name']]
lats = [closest_substation[circuit][3] for circuit in data['circuit_name']]
data['zip_code'] = zips
data['longitude'] = longs
data['latitude'] = lats
data.info()


# In[11]:

# Write to temporary file for editing outside repo
data.to_csv('../data/processed/temp-shutoffs.csv')
# Write missing data to missing data file
data[data.zip_code.isnull()].to_csv('../data/processed/missing-zips.csv')

