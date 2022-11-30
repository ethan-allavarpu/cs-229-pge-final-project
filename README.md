# Out For How Long? Predicting the Severity of Planned Power Shutoffs

In recent years, California has seen an uptick in wildfire frequency and severity as a result of climate change. Pacific Gas & Electric (PG&E) has had many lawsuits filed against them because of the role their power lines play in contributing to wildfire events. As a result, PG&E began to implement Public Safety Power Shutoff (PSPS) events to reduce the likelihood of their starting or contributing to wildfire events. These events, however, disrupt lives. We want to predict the severity of these events (with  shutoff length as a proxy) based on weather conditions (e.g., temperature, wind), geographic location, and census data (e.g., population, income) given that PG&E planned a shutoff.

## Model Performance

Updated as of _November 30, 2022_

| Model                     |Test R-Squared|RMSE           |MAE            |
|:--------------------------|-------------:|--------------:|--------------:|
|Simple Linear Regression   |   0.002391   |  1578.099447  |  1157.988578  |
|Multiple Linear Regression |   0.465563   |  1155.053925  |   882.595626  |
|XGBoost (Preliminary)      |   0.576786   |  1027.860530  |   750.430526  |
|Ridge                      |   0.530090   |  1083.081942  |   826.343620  |
|LASSO                      |   0.526052   |  1087.725956  |   826.844061  |
|Elastic Net                |   0.526052   |  1087.725956  |   826.844061  |
|**Random Forest**          | **0.766858** | **762.893741**| **500.366251**|
|KNN                        |   0.726484   |   826.315377  |   544.658185  |
|**XGBoost**                | **0.768380** | **760.399548**| **519.912074**|
|Neural Network             |   0.615241   |   831.422925  |   583.833951  |

The best model(s) is (are) listed in **bold**.

The models are listed in the order in which they were created (earliest to latest). We note that, when performing cross-validation, the Elastic Net ended up selected the same model as LASSO.

## Additional Links

- Note that the census data is too large to house in the repository
  - Population and Demographics: <https://data.census.gov/cedsci/table?q=zcta&tid=ACSDP5Y2020.DP05>
  - Income: <https://data.census.gov/cedsci/table?q=income%20by%20zcta&tid=ACSST5Y2020.S1901>
- Substation Locations: <https://cecgis-caenergy.opendata.arcgis.com/datasets/7f37f2535d3144e898a53b9385737ee0_0/explore?location=39.055127%2C-122.040695%2C11.00>
