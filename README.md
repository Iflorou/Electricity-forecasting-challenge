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
├── data/                             (not tracked — add your parquet files here)
│   ├── target_train.parquet          (training targets)
│   ├── weather_train.parquet         (training weather)
│   ├── network_train.parquet         (training network data)
│   ├── weather_test.parquet          (2025 weather)
│   └── network_test.parquet          (2025 network data)
│
├── models/
│   ├── xgboost_solar_model.pkl       (trained solar model — XGBoost)
│   ├── tuned_lgbm_wind_model.pkl     (trained wind model — LightGBM)
│   └── load_final.pkl                (trained load model — LightGBM)
│
├── predictions/
│   ├── FR_solar_predicted_2025.parquet
│   ├── FR_wind_predicted_2025.parquet
│   └── FR_load_predicted_2025.parquet
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
  Log transformations for skewed distributions

Time features (cyclical encoding - critical!):
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
-  EEX_COAL, EEX_GAS_PEG (fuel costs affect consumption indirectly)
-  FR_availability_coal, gas, hydro, nuclear (what's available affects system)

**Price Model** (all features relevant):
-  Fuel prices (direct generation cost signal)
-  Capacity & availability (supply constraints)

---




## Model Performance

### Solar Model
```
Training:
  RMSE: 137 MW
  R²: 0.997

Validation (2024):
  RMSE: 1,493 MW
  MAE: 833 MW
  WMAPE: 33.7%
  R²: 0.824

Cross-Validation (TimeSeriesSplit):
  Fold 1 (2020-early 22): R² = 0.933
  Fold 2 (2020-mid 23):   R² = 0.893
  Fold 3 (2020-2023):     R² = 0.827
  Average: R² = 0.884

Interpretation:
- Degradation from fold 3 to validation expected
- Cause: Solar capacity grew 30% in 2024
- Model generalizes well despite distribution shift
```

### Wind Model
```
Validation (2024):
  RMSE: 1,404 MW
  MAE: 959 MW
  WMAPE: 18.4%
  R²: 0.872

Why better than solar:
- Wind never goes to zero (no division errors in WMAPE)
- More stable patterns (less weather-dependent)
- Better at capturing temporal dynamics
```

### Load Model
```
Validation (2024):
  RMSE: 2,587 MW
  MAE: 1,938 MW
  WMAPE: 3.96%
  MAPE: 4.00%
  R²: 0.929

Why excellent:
- Temperature is strong causal signal (heating/cooling)
- Load never near zero (range: 20,000-60,000 MW)
- Consistent daily/weekly patterns
- Lowest relative error (4.3% of max value)
```

### Price Model
```
Validation (2024):
  RMSE: ~8-12 €/MWh
  WMAPE: ~15-25%
  MAPE: ~15-25%
  R²: ~0.70-0.85

Challenges:
- Market dynamics (speculation, bidding strategies)
- Depends on other models' predictions (error propagation)
- Many external factors not in data
```

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



## Setup & Usage

### Requirements

```
Python 3.8+
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 0.24.0
lightgbm >= 3.2.0
xgboost >= 1.4.0
matplotlib >= 3.4.0
jupyter >= 1.0.0
```

### Installation

```bash
# Clone repository
git clone https://github.com/Iflorou/Electricity-forecasting-challenge.git
cd Electricity-forecasting-challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```






## Output Files

### Predictions (Main Deliverable)

**Prediction files (Apache Parquet, hourly 2025)**
```
predictions/
├── FR_solar_predicted_2025.parquet   Solar generation [MW]
├── FR_wind_predicted_2025.parquet    Wind generation [MW]
└── FR_load_predicted_2025.parquet    Load [MW]



### Models (Trained Artifacts)

```
models/
├── xgboost_solar_model.pkl       (XGBoost)
├── tuned_lgbm_wind_model.pkl     (LightGBM, tuned)
└── load_final.pkl                (LightGBM)
```




