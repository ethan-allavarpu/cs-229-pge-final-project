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


substations = pd.read_csv('../data/raw/California_Electric_Substations.csv',
                          usecols=['Name', 'ZIP_CODE', 'COUNTY', 'Lon', 'Lat'], dtype='str')
# Standardize names for dropping
substations.Name = [stn.upper() for stn in substations.Name]
# Don't include unknown names or ones that start with numbers
substations = substations[substations.Name != 'UNKNOWN'].reset_index(drop=True)
substations = substations[
    [re.match('^\d', name) is None for name in substations.Name]
]
substations


# In[3]:


def clean_df(psps, start_datetype, end_datetype):
    """
    Clean and standardize data frame inputs so they can later be joined.
    Standardize column names, column types and imputation (if necessary)
    """
    # Standardize columns, column names
    if 'Distribution / Transmission' in psps.columns:
        psps['Distribution / Transmission'] = [
            x.upper() for x in psps['Distribution / Transmission']
        ]
        psps = psps[psps['Distribution / Transmission'] == 'DISTRIBUTION']
        psps.drop(columns='Distribution / Transmission', inplace=True)
    # Clean circuit name
    cleaned_circuit = [
        re.findall('.*?(?=\s\d{4,}\*?)', str(circuit))[0].strip()
        if len(re.findall('.*?(?=\s\d{4,}\*?)', str(circuit))) > 0 else circuit
        for circuit in psps['Circuit Name']
    ]
    psps['Circuit Name'] = cleaned_circuit
    # Convert fire risk to integer
    cleaned_hftd = [
        int(max(re.findall('\d', str(hftd))))
        if len(re.findall('\d', str(hftd))) > 0 else 0
        for hftd in psps['HFTD Tier(s)']
    ]
    psps['HFTD Tier'] = cleaned_hftd
    psps.columns = [str(col_name).strip() for col_name in psps.columns]
    # Shorten column names
    shorter_names = [
        re.sub(' Customers$', '', str(col_name)) for col_name in psps.columns
    ]
    psps.columns = shorter_names
    # Standardize column names further
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
    # Extract datetime forms
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
    # Get length of shutoff
    psps['time_out_min'] = (
        (psps.restoration_time - psps.deenergize_time) / pd.Timedelta('1m')
    )
    psps['Key Communities'] = fixed_communities
    # Convert columns to floats
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
    # Fill in missingness with means for same circuit
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
    # Missing if too early datetime variable
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


# Cleaning dataframe simple for 2020-2021 data frames
def clean_df_20_21(file_path, start_datetype, end_datetype):
    """
    Read in and clean dataframes from 2020-2021
    """
    psps = pd.read_csv(file_path, dtype=str)
    return clean_df(psps, start_datetype, end_datetype)
# Need to join two separate dataframes for 2018-2019 instances
def clean_df_18_19(psps_date, start_datetype, end_datetype):
    """
    Join separate data frames in 2018-2019 to link customers and circuits
    before cleaning the data frame
    """
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
    # Join on circuits between customer and outage data
    psps = pd.merge(circuits, customers, how='outer', on='Circuit Name')
    psps = psps[[
        re.search('LINE$', circuit) is None
        for circuit in psps['Circuit Name']
    ]]
    return clean_df(psps, start_datetype, end_datetype)


# In[5]:


# Datetime format for 2018-2019
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

# Combine 2018-2019 outages into single dataframe
data_18_19 = pd.concat(
    [clean_df_18_19(file[0], file[1], file[2]) for file in data_input],
    axis=0
).reset_index(drop=True)


# In[6]:


# Datetime format for 2020-2021 outages
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

# Combine 2020-2021 outages
data_20_21 = pd.concat(
    [clean_df_20_21(file[0], file[1], file[2]) for file in data_input],
    axis=0
).reset_index(drop=True)


# In[7]:


data = pd.concat([data_18_19, data_20_21], axis=0)
# Remove missing data
data = data[
        (data.restoration_time > '2000-01-01') &
        (data.deenergize_time > '2000-01-01') &
        (~data.total_affected.isnull()) &
        (~data.total_affected.isna())
    ].reset_index(drop=True)


# In[8]:


data


# In[9]:


# Fuzzymatching to link to substation
def most_similar_station(circuit, stns, thresh=80):
    """
    Return the matching substation for a given circuit
    Fuzzy-matching to see similarity scores (minimum of thresh)
    
    circuit is circuit name, stns is dataframe of stations, thresh is minimum
    value for matching
    """
    circuit = re.sub('\.', '', circuit)
    circuit = re.sub(' NO', '', circuit)
    circuit = re.sub('\d', '', circuit).strip()
    # Use fuzzymatching ratio to look for most similar
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
    # Need most similar score to be at least of value thresh (80%)
    if max_score < thresh:
        return ('Default', None, None, None)
    return [scores[1:] for scores in sim_scores if scores[0] == max_score][0]


# In[10]:


# This cell is the longest to run (about 5 minutes)
unique_circuits = data['circuit_name'].unique()
closest_substation = {
    circuit: most_similar_station(circuit, substations)
    for circuit in unique_circuits
}


# In[11]:


# Add substation data (location) to dataframe
zips = [closest_substation[circuit][1] for circuit in data['circuit_name']]
longs = [closest_substation[circuit][2] for circuit in data['circuit_name']]
lats = [closest_substation[circuit][3] for circuit in data['circuit_name']]
data['zip_code'] = zips
data['longitude'] = longs
data['latitude'] = lats
data.info()


# In[12]:


data.to_csv('../data/processed/temp-shutoffs.csv')
# Write missing data to missing data file
data[data.zip_code.isnull()].to_csv('../data/processed/missing-zips.csv')

