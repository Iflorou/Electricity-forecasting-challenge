# Electricity Forecasting Challenge - Kpler Power & Gas

> Time-series machine learning models for hourly electricity forecasting in France (2025)

**4 Targets:** Load • Solar Generation • Wind Generation • Electricity Price

---

## Project Overview

This project builds machine learning models to forecast hourly electricity metrics for France in 2025 across four critical dimensions:

- **Solar Generation** Solar PV electricity production (MW)
- **Wind Generation** Wind turbine electricity production (MW)
- **Load**  Electricity demand/consumption (MW)
- **Price**   Day-ahead market electricity price (€/MWh)

The models use weather data (312 geographic tiles with 4 variables), network infrastructure data (capacity, availability, fuel prices), and temporal patterns to make production-ready predictions.

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Reproducing the predictions

All trained models and intermediate feature files are already committed, so you can run any notebook independently. To reproduce everything from scratch, run the notebooks in order:

| Notebook | What it does |
|---|---|
| `01_kpler_EDA.ipynb` | Exploratory data analysis (optional, no outputs saved) |
| `02_kpler_solar.ipynb` | Trains solar model → saves `models/xgboost_solar_model.pkl` + feature parquets |
| `03_kpler_wind.ipynb` | Trains wind model → saves `models/tuned_lgbm_wind_model.pkl` + feature parquets |
| `04_kpler_load.ipynb` | Trains load model → saves `models/load_final.pkl` + feature parquets |
| `05_kpler_price.ipynb` | Trains price model using outputs of 02–04 as stacked features → saves final predictions |

**To only reproduce the test predictions without retraining:** open `05_kpler_price.ipynb` directly — it loads the pre-trained models and saves the four prediction parquets to `predictions/`.

---

## Data Description

### Training Data (2020-2023)

```
target_train.parquet
├─ FR_load_actual: Hourly electricity load (MW)
├─ FR_solar_actual: Hourly solar generation (MW)
├─ FR_wind_actual: Hourly wind generation (MW)
└─ FR_price_actual: Day-ahead market price (€/MWh)

weather_train.parquet
├─ 312 geographic tiles × 4 hourly variables:
│  ├─ SSRD: Solar surface radiation (J/m²)
│  ├─ TCC: Total cloud cover (0-1)
│  ├─ 2T: Temperature (K)
│  └─ 100WS: Wind speed at 100m (m/s)
└─ Time period: 2020-2023, hourly (35,040 samples)

network_train.parquet
├─ Fuel prices:
│  ├─ EEX_CARBON: Carbon cost (€/tonne)
│  ├─ EEX_COAL: Coal cost (€/MWh)
│  └─ EEX_GAS_PEG: Gas cost (€/MWh)
├─ Generation capacity (MW):
│  ├─ FR_capacity_solar
│  ├─ FR_capacity_wind
│  ├─ FR_capacity_coal
│  ├─ FR_capacity_gas
│  ├─ FR_capacity_hydro
│  └─ FR_capacity_nuclear
└─ Availability (% or MW):
   ├─ FR_availability_coal
   ├─ FR_availability_gas
   ├─ FR_availability_hydro
   └─ FR_availability_nuclear
```


## Project Structure

```
Electricity-forecasting-challenge/
├── README.md                          (this file)
├── notebooks/
│   ├── 01_kpler_EDA.ipynb            (exploratory data analysis)
│   ├── 02_kpler_solar.ipynb          (solar generation forecasting)
│   ├── 03_kpler_wind.ipynb           (wind generation forecasting)
│   ├── 04_kpler_load.ipynb           (electricity load forecasting)
│   └── 05_kpler_price.ipynb          (electricity price forecasting)
│
├── data/
│   ├── target_train.parquet          (training targets)
│   ├── weather_train.parquet         (training weather)
│   ├── network_train.parquet         (training network data)
│   ├── weather_test.parquet          (2025 weather features)
│   ├── network_test.parquet          (2025 network features)
│   └── X_train_*.parquet / X_valid_*.parquet / X_test_*.parquet  (pre-built feature sets per target)
│
├── models/
│   ├── xgboost_solar_model.pkl       (trained solar model — XGBoost)
│   ├── tuned_lgbm_wind_model.pkl     (trained wind model — LightGBM)
│   └── load_final.pkl                (trained load model — XGBoost)
|   └── price_final.pkl               (trained price model — LightGBM)
    
│
├── predictions/
│   ├── FR_solar_predicted_2025.parquet
│   ├── FR_wind_predicted_2025.parquet
│   └── FR_load_predicted_2025.parquet
|   └── FR_price_predicted_2025.parquet
│
└── requirements.txt                  (Python dependencies)
```

---

## Feature Engineering

### Weather Features (~20 features)
```
Raw aggregation (312 tiles → mean & std):
   SSRD_mean, SSRD_std
   TCC_mean, TCC_std
   Temp_mean, Temp_std
   Wind_mean, Wind_std

Temporal dependencies (lags & rolling):
  Lag 1h, Lag 24h for each weather variable
  Rolling windows: 3h, 6h, 24h (mean & std)
  Log transformations for skewed distributions (SSRD)

Time features (cyclical encoding ):
  hour_sin, hour_cos (captures daily patterns, 0-24h cycle)
  dow_sin, dow_cos (captures weekly patterns, Monday-Sunday)
  month_sin, month_cos (captures seasonality, 1-12 months)
```

### Network Features

**Solar Model** (only causal features):
-  FR_capacity_solar (more panels = more generation)

**Wind Model** (only causal features):
-  FR_capacity_wind (more turbines = more generation)

**Load Model** (causal & indirect features):
-  EEX_COAL, EEX_GAS_PEG 
-  FR_availability_coal, gas, hydro, nuclear 

**Price Model** (all features relevant):
-  Fuel prices (direct generation cost signal)
-  Capacity & availability (supply constraints)

---

## Validation Strategy

### Metrics Used

**RMSE** (Root Mean Squared Error)
- Penalizes large errors
- Units: MW or €/MWh
- Primary metric for this challenge

**MAE** (Mean Absolute Error)
- Interpretable, same units as target
- Less sensitive to outliers than RMSE

**WMAPE** (Weighted Mean Absolute Percentage Error)
- Weighted by actual values
- Better than MAPE for values near zero (solar at night)
- Safe for all targets including solar

**MAPE** (Mean Absolute Percentage Error)
- Percentage error
- Use for load & price (no near-zero values)
- Avoid for solar/wind (unreliable with zeros)

**R²** (Coefficient of Determination)
- Variance explained
- 0 = poor, 1.0 = perfect
- Comparable across targets









