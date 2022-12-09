# Python script taken from scrape_weather.ipynb

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly
import numpy as np
import pandas as pd
import geopy.distance
import glob

# Load processed data to get substation information
data = pd.read_csv("../processed/processed-shutoffs.csv")
data["deenergize_time"] = pd.to_datetime(data["deenergize_time"])
data["restoration_time"] = pd.to_datetime(data["restoration_time"])
data["deenergize_date"] = data["deenergize_time"].dt.date
data["restoration_date"] = data["restoration_time"].dt.date

# Fetch data from Meteostat
missing = {}
full_data = {}
weather_present = {}
data_missing = {}
daily = {}
daily_data_missing = {}
for i in range(data.shape[0]):
    point_1 = Point(data.loc[i, "latitude"], data.loc[i, "longitude"])
    start = pd.to_datetime(data.loc[i, "deenergize_date"]) - timedelta(days=5)
    end = pd.to_datetime(data.loc[i, "deenergize_date"])
    # weather = Hourly(point_1, start, end)
    weather = Daily(point_1, start, end)
    weather = weather.fetch()
    # if weather.shape[0] == 0:
    #     weather = Daily(point_1, start, end)
    #     weather = weather.fetch()
    if weather.shape[0] == 0:
        missing[i] = data.loc[i, ["circuit_name",
                                  "deenergize_time", "latitude", "longitude"]]
    #     daily[i] = weather
    elif weather.shape[0] >= 5:
        if weather.shape[0] > 5:
            weather = weather.loc[start.date(
            ): end.date() - timedelta(days=1), :]
            assert weather.shape[0] == 5
        if sum(np.sum(weather[["tmin", "tmax", "wspd"]].isna(), axis=1)) == 0:
            full_data[i] = data.loc[i, ["circuit_name",
                                        "deenergize_time", "latitude", "longitude"]]
            weather["circuit_name"] = data.loc[i, "circuit_name"]
            weather["deenergize_time"] = data.loc[i, "deenergize_time"]
            weather["source_circuit_name"] = data.loc[i, "circuit_name"]
            weather["source_deenergize_time"] = data.loc[i, "deenergize_time"]
            weather["approximated"] = False
            daily[i] = weather
        else:
            data_missing[i] = data.loc[i, ["circuit_name",
                                           "deenergize_time", "latitude", "longitude"]]
            weather["circuit_name"] = data.loc[i, "circuit_name"]
            weather["deenergize_time"] = data.loc[i, "deenergize_time"]
            weather["source_circuit_name"] = data.loc[i, "circuit_name"]
            weather["source_deenergize_time"] = data.loc[i, "deenergize_time"]
            weather["approximated"] = False
            daily_data_missing[i] = weather
    else:
        weather_present[i] = data.loc[i, ["circuit_name",
                                          "deenergize_time", "latitude", "longitude"]]

# Track substations with missing/partially missing/full data
full_data = pd.DataFrame(full_data).T
full_data.to_csv("full_weather.csv")
missing = pd.DataFrame(missing).T
missing.to_csv("missing_weather.csv")
weather_present = pd.DataFrame(weather_present).T
weather_present.to_csv("some_missing_weather.csv")
data_missing = pd.DataFrame(data_missing).T
data_missing.to_csv("data_missing.csv")

# Save stations with full weather
for i in list(daily.keys()):
    shutoff = data.iloc[i, :]
    filename = f"raw/daily/weather_{i}_{shutoff['circuit_name']}_{shutoff['deenergize_date']}.csv"
    weather = daily[i]
    # assert os.path.isfile(filename) == False
    weather.to_csv(filename)

# Entirely missing:
# Replace stations that are entirely missing weather data with a copy
# of their nearest neighbor's weather

knn_missing = {}
to_drop = {}
for i in missing.index:
    de_time = missing.loc[i, "deenergize_time"].date()
    dist = full_data[full_data["deenergize_time"].dt.date == de_time].apply(lambda row: geopy.distance.geodesic(
        (row["latitude"], row["longitude"]),
        (missing.loc[i, "latitude"],
         missing.loc[i, "longitude"])).miles,
        axis=1)
    if len(dist) == 0:
        to_drop[i] = missing.loc[i]
        continue
    min_idx = dist.idxmin()
    dist = min(dist)
    if dist > 60:
        to_drop[i] = missing.loc[i]
        continue
    knn_missing[i] = {
        "distance": dist,
        "min_idx": min_idx,
        "closest_circuit_name": full_data.loc[min_idx].loc["circuit_name"],
        "closest_deenergize_time": full_data.loc[min_idx].loc["deenergize_time"],
        "closest_latitude": full_data.loc[min_idx].loc["latitude"],
        "closest_longitude": full_data.loc[min_idx].loc["longitude"]
    }

# Join to get rest for information
knn_missing = pd.DataFrame(knn_missing).T.join(missing)

# Create copy of weather from nearest neighbor, write to disk
for i in knn_missing.index:
    file_name = glob.glob(
        f"raw/daily/weather_{knn_missing['min_idx'].loc[i]}_{knn_missing['closest_circuit_name'].loc[i]}_{str(knn_missing['closest_deenergize_time'].loc[i].date())}.csv")
    assert len(file_name) == 1
    curr_weather = pd.read_csv(file_name[0])
    curr_weather["circuit_name"] = knn_missing['circuit_name'].loc[i]
    curr_weather["deenergize_time"] = knn_missing['deenergize_time'].loc[i]
    curr_weather["source_circuit_name"] = knn_missing['closest_circuit_name'].loc[i]
    curr_weather["source_deenergize_time"] = knn_missing['closest_deenergize_time'].loc[i]
    curr_weather["approximated"] = True
    curr_weather = curr_weather.set_index("time")
    curr_weather.to_csv(
        f"raw/daily/weather_{i}_{knn_missing['circuit_name'].loc[i]}_{str(knn_missing['deenergize_time'].loc[i].date())}.csv")

# Partially missing:
# Replace stations that are missing some rows of weather with:
# a copy of their nearest neighbor's weather

knn_some_missing = {}
for i in weather_present.index:
    de_time = weather_present.loc[i, "deenergize_time"].date()
    dist = full_data[full_data["deenergize_time"].dt.date == de_time].apply(lambda row: geopy.distance.geodesic(
        (row["latitude"], row["longitude"]),
        (weather_present.loc[i, "latitude"],
         weather_present.loc[i, "longitude"])).miles,
        axis=1)
    if len(dist) == 0:
        to_drop[i] = weather_present.loc[i]
        continue
    min_idx = dist.idxmin()
    dist = min(dist)
    if dist > 60:
        to_drop[i] = weather_present.loc[i]
        continue
    knn_some_missing[i] = {
        "distance": dist,
        "min_idx": min_idx,
        "closest_circuit_name": full_data.loc[min_idx].loc["circuit_name"],
        "closest_deenergize_time": full_data.loc[min_idx].loc["deenergize_time"],
        "closest_latitude": full_data.loc[min_idx].loc["latitude"],
        "closest_longitude": full_data.loc[min_idx].loc["longitude"]
    }

# Join to get rest for information
knn_some_missing = pd.DataFrame(knn_some_missing).T.join(weather_present)

# Create copy of weather from nearest neighbor, write to disk
for i in knn_some_missing.index:
    file_name = glob.glob(
        f"raw/daily/weather_{knn_some_missing['min_idx'].loc[i]}_{knn_some_missing['closest_circuit_name'].loc[i]}_{str(knn_some_missing['closest_deenergize_time'].loc[i].date())}.csv")
    assert len(file_name) == 1
    curr_weather = pd.read_csv(file_name[0])
    curr_weather["circuit_name"] = knn_some_missing['circuit_name'].loc[i]
    curr_weather["deenergize_time"] = knn_some_missing['deenergize_time'].loc[i]
    curr_weather["source_circuit_name"] = knn_some_missing['closest_circuit_name'].loc[i]
    curr_weather["source_deenergize_time"] = knn_some_missing['closest_deenergize_time'].loc[i]
    curr_weather["approximated"] = True
    curr_weather = curr_weather.set_index("time")
    curr_weather.to_csv(
        f"raw/daily/weather_{i}_{knn_some_missing['circuit_name'].loc[i]}_{str(knn_some_missing['deenergize_time'].loc[i].date())}.csv")

# Drop stations where nearest station is too far
pd.DataFrame(to_drop).T.to_csv("to_drop.csv")

# All days present, but missing at least one of tmin, tmax, wspd. Replace with
# a copy of their nearest neighbor's data
knn_data_missing = {}
for i in data_missing.index:
    de_time = data_missing.loc[i, "deenergize_time"].date()
    dist = full_data[full_data["deenergize_time"].dt.date == de_time].apply(lambda row: geopy.distance.geodesic(
        (row["latitude"], row["longitude"]),
        (data_missing.loc[i, "latitude"],
         data_missing.loc[i, "longitude"])).miles,
        axis=1)
    if len(dist) == 0:
        to_drop[i] = data_missing.loc[i]
        continue
    min_idx = dist.idxmin()
    dist = min(dist)
    if dist > 60:
        to_drop[i] = data_missing.loc[i]
        continue
    knn_data_missing[i] = {
        "distance": dist,
        "min_idx": min_idx,
        "closest_circuit_name": full_data.loc[min_idx].loc["circuit_name"],
        "closest_deenergize_time": full_data.loc[min_idx].loc["deenergize_time"],
        "closest_latitude": full_data.loc[min_idx].loc["latitude"],
        "closest_longitude": full_data.loc[min_idx].loc["longitude"]
    }
    daily_data_missing[i].loc[:, ["tmin", "tmax", "wspd"]] = daily_data_missing[i].loc[:, ["tmin", "tmax", "wspd"]].reset_index().fillna(
        daily[min_idx][["tmin", "tmax", "wspd"]].reset_index()).set_index("time")
    daily_data_missing[i]["approximated"] = True
    daily_data_missing[i].to_csv(
        f"raw/daily/weather_{i}_{data_missing['circuit_name'].loc[i]}_{str(data_missing['deenergize_time'].loc[i].date())}.csv")
