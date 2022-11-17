# cs229-pge-final-project

## Model Performance
Updated as of _November 17, 2022_

Best model in **bold** (currently, Random Forest):

| Model                     |Test R-Squared|RMSE          |
|:--------------------------|-------------:|-------------:|
|Simple Linear Regression   |	0.002391     |  1578.099447 |
|Multiple Linear Regression |	0.465563     |  1155.053925 |
|XGBoost (Preliminary)      |	0.576786     |  1027.860530 |
|Ridge                      |	0.487803     |  1130.766247 |
|LASSO                      |	0.489623     |  1128.755645 |
|Elastic Net                |	0.481117     |  1138.122057 |
|**Random Forest**          | **0.767109** |**762.484073**|
|KNN                        | 0.726484     |   826.315377 |


## Additional Links

  - Note that the census data is too large to house in the repository
    - Population and Demographics: https://data.census.gov/cedsci/table?q=zcta&tid=ACSDP5Y2020.DP05
    - Income: https://data.census.gov/cedsci/table?q=income%20by%20zcta&tid=ACSST5Y2020.S1901
  - Substation Locations: https://cecgis-caenergy.opendata.arcgis.com/datasets/7f37f2535d3144e898a53b9385737ee0_0/explore?location=39.055127%2C-122.040695%2C11.00
