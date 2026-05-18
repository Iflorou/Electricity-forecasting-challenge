# Data Scientist (DS I, II, III) Interview Case Study - Kpler Power & Gas crew

## 1) Your task

You are a data scientist at Kpler, in the Power business unit.

You are going to build ML models in order to predict the following targets, at hourly granuarity:
- `FR_load_actual`  in MW (total power demand on the FR electricity transport grid)
- `FR_solar_actual` in MW (total output of grid-connected PV power plants, excluding distributed PV such as rooftops etc..)
- `FR_wind_actual`  in MW (total output of offshore + onshore wind turbines, excluding distributed systems)
- `FR_price_actual` in €/MWh (price in the spot electricity market)

We'll provide you with
- training features, covering jan 2020 to dec 2025
- targets values, covering jan 2020 to dec 2024
👉 Your goal is to predict these targets over 2025

It is up to you to decide on the best way to get there, with the features provided to you below
However, it is forbidden to try to access other external data sources for this exercise


### Deliverables
- Your hourly predictions for each targets in 2025, in parquet format
  - We will compute your performance on our held-out test set
- A self performance evaluation of your model and approach
  - With the performance metrics and methodology of your choice that you deem meaningful for the task
- The code you used to generate these forecasts
  - Minimal requirement: A python code that can be run by your reviewer (python package, or notebook, etc...) that:
    - allow to reproduces your test result parquet
    - is self explanatory
  - Optional requirement: A python package that structures your code in a way that
    - allow us to run an inference easily from any new future test set (e.g. 2026...)
    - can scale to other geographical zones if in the future we start having more than FR in our training set

Be ready to explain your methodology and code in an upcoming interview.
Feel free to use tools of your choice (including LLMs) but be ready to answer openly about which tool you used and where in your code precisely.
Anticipate how complex might be the task, and take any direction you deem necessary to allow yourself to finish on time: forecast accuracy will only play a part in our overall evaluation.

## 2) Data
- `data/` contains data files at your disposal to solve your problem

### 2.1) `target_train.parquet` (2020-2024)
Here are your targets, at hourly granularity, from jan 2020 to dec 2024, for training purposes, as parquet format

They consist in:
- `FR_load_actual`  in MW
- `FR_solar_actual` in MW
- `FR_wind_actual`  in MW
- `FR_price_actual` in €/MWh

You'll be tasked with predicting them for the year 2025

### 2.2) `weather_train.parquet` (2020-2024) and `weather_test.parquet` (2025)

Parquet file containing historical weather data at hourly granularity, from 2020 to 2024 for the train, and 2025 for the test.


| start_date                |   ('FR', '006d9e54d8b83baf86c443cba26fe066', 'ssrd') |   ('FR', '006d9e54d8b83baf86c443cba26fe066', 'tcc') |
|:--------------------------|-----------------------------------------------------:|----------------------------------------------------:|
| 2020-01-01 10:00:00+00:00 |                                               482445 |                                            0.640137 |
| 2020-01-01 11:00:00+00:00 |                                               661939 |                                            0.569824 |
| 2020-01-01 12:00:00+00:00 |                                               912383 |                                            0.763184 |


The columns are a MultiIndex: `(zone, tile_id, weather_variable)`, where:

- `zone` is one of `{"FR"}`
- `weather_variable` is one of:
  - `tcc`   # Total cloud cover (0-1)
  - `2t`    # 2m temperature (Celsius)
  - `ssrd`  # Surface solar radiation downwards (J/m^2)
  - `100ws` # 100m wind speed (m/s)
- `tile_id` represent a unique geographical square of 50*50km
  - these tiles are covering a good portion of their respective country and are each disjoint, so that overall they are representative of the country's weather (there are 312 tiles for France in your dataset)
  - think of these tiles as 'simplification' of otherwise very granular weather data: weather_variables are averaged inside each tiles and we only gave you their averaged values
  - not all weather variable are present for all tiles

So, overall, `('FR', '00a5ba9b5c6956348da64608421b52f2', 'ssrd')` represent the average solar irradiance in the tile '00a5ba9...' that lies somewhere in FR.

### 2.3) `network_train.parquet` (2020-2024) and `network_test.parquet` (2025)

Hourly power grid network data, containing the following features:

- Fuel prices
  - `EEX_CARBON`: EU-ETS CO2 price in €/tCO2
  - `EEX_COAL`: Coal prices in €/MWh (thermal MWh, not electric one)
  - `EEX_GAS_PEG`: Natural gas prices at PEG hub, in €/MWh (thermal MWh, not electric one)
- Capacity installed in MW
  - `FR_capacity_solar`
  - `FR_capacity_wind`
- Availabilities of power plant type in MW, which can be inferior to capacity installed (in case of maintenance for instance)
  - `FR_availability_coal`
  - `FR_availability_gas`
  - `FR_availability_hydro`
  - `FR_availability_nuclear`

## 3) Modeling hints
You don't have to be an expert in power grid to pass this test, so we give you some hints below

- electricity demand is generally influenced by many things: put yourself in the shoes of a typical electricity consumer here!
  - external temperature
  - time of day
  - ...

- electricity price (for a given target timestamp) is the result from supply and demands bids for that target time. It is therefore influenced by
  - total electricity demand
  - total electricity supplied by fatal energy sources (wind and solar, which have 0 fuel costs)
  - and the price at which conventional power plants (coal, gas, hydro, nuclear) are willing to produce

With that in mind, you should have a better idea into how to best answer our problem!
