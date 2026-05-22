

## EVALUATION FRAMEWORK

### Metrics Selected
1. **R² Score** - Explains variance in target (primary metric)
2. **RMSE** - Root Mean Squared Error (penalizes large errors)
3. **WMAPE** - Weighted Mean Absolute Percentage Error (percentage accuracy)
4. **MAE** - Mean Absolute Error (interpretable in original units)
5. **Generalization Gap** - (Training - Validation) shows overfitting


---

##  MODEL 1: SOLAR FORECASTING

**Architecture:** XGBoost Regressor  

**Features:** 28 (weather + time + capacity)

| Group | Features |
|---|---|
| Weather aggregates (6) | `tcc_mean`, `tcc_std`, `temp_mean`, `temp_std`, `wind_mean`, `wind_std` |
| SSRD (4) | `ssrd_log`, `ssrd_std_log`, `ssrd_lag1`, `ssrd_lag24` |
| Rolling stats (11) | `ssrd_roll_3h`, `ssrd_roll_24h`, `tcc_roll_3/24h_mean`, `tcc_roll_6h_std`, `temp_roll_3/24h_mean`, `temp_roll_6h_std`, `wind_roll_3/24h_mean`, `wind_roll_6h_std` |
| Time encoding (4) | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Solar-specific (2) | `ssrd_norm`, `solar_potential` |
| Capacity (1) | `FR_capacity_solar` |

### Performance Metrics

| Metric | Value | 
|--------|-------|
| **Training R²** | 0.9940|
| **Validation R²** | 0.8318|


- **High training R²** (0.99): Model learns underlying patterns well
- **Reasonable RMSE** (1,458 MW): ~10% of peak generation
- **Poor WMAPE** (32.44%): ~1 in 3 predictions off by >32%
- **Significant overfitting**: 16.2 pp gap (training 0.99 → validation 0.83)


---
##  MODEL 2: WIND FORECASTING

**Architecture:** LightGBM Regressor (Tuned)  

**Features:** 22 (weather + time + capacity)

| Group | Features |
|---|---|
| Weather aggregates (8) | `tcc_mean`, `tcc_std`, `temp_mean`, `temp_std`, `wind_mean`, `wind_std`, `ssrd_log`, `ssrd_std_log` |
| Rolling stats (9) | `tcc_roll_3/24h_mean`, `tcc_roll_6h_std`, `temp_roll_3/24h_mean`, `temp_roll_6h_std`, `wind_roll_3/24h_mean`, `wind_roll_6h_std` |
| Time encoding (4) | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Capacity (1) | `FR_capacity_wind` |

### Performance Metrics

| Metric | Value | 
|--------|-------|
| **Training R²** | 0.9949 | 
| **Validation R²** | 0.8959 | 

- **Best R² after tuning** (0.896): Among best in submission
- **High WMAPE** (16.15%): 1 in 6 predictions off by >16%
- **Low overfitting** (9.9 pp gap): Model generalizes well
- **Training-validation gap still exists** (9.9 pp): Room for improvement




## MODEL 3: LOAD FORECASTING

**Architecture:** XGBOOST Regressor  
**Features:** 20 (temperature + time + network)

| Group | Features |
|---|---|
| Temperature (8) | `temp_mean`, `temp_std`, `temp_lag1`, `temp_lag24`, `temp_roll_3h`, `temp_roll_6h`, `temp_roll_24h`, `temp_roll_3h_std` |
| Time encoding (6) | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos` |
| Fuel prices (2) | `EEX_COAL`, `EEX_GAS_PEG` |
| Grid availability (4) | `FR_availability_coal`, `FR_availability_gas`, `FR_availability_hydro`, `FR_availability_nuclear` |

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training R²** | 0.9949|
| **Validation R²** | 0.8959|



- **High WMAPE** (4.10%): 25:1 ratio for correct predictions
- **High R²** (0.918): Best R² in entire submission!
- **Minimal overfitting** (6.1 pp): Strong generalization




---

## MODEL 4: PRICE FORECASTING

**Architecture:** LightGBM Regressor with Stacking (OOF predictions as features)  
**Features:** 23 (20 base + 3 stacked predictions)

| Group | Features |
|---|---|
| Weather (9) | `temp_mean`, `temp_std`, `temp_roll_3/24h_mean`, `tcc_mean`, `tcc_std`, `tcc_roll_3/24h_mean`, `tcc_roll_6h_std` |
| Time encoding (4) | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Fuel prices (3) | `EEX_CARBON`, `EEX_COAL`, `EEX_GAS_PEG` |
| Grid availability (4) | `FR_availability_coal`, `FR_availability_gas`, `FR_availability_hydro`, `FR_availability_nuclear` |
| Stacked predictions (3) | `pred_solar`, `pred_wind`, `pred_load` (OOF predictions from models 1–3) |


A substantial drop in performance from training data (R²: 0.98, MAE: 10.09) tovalidation data (R²: 0.1533, MAE: 29.94). This usually indicates that the model is overfitting to the training data.

Overfitting means the model has learned the training data too well, including its noise and specific patterns, and thus struggles to generalize to new, unseen data (like your validation set).

This model needs care!! 
