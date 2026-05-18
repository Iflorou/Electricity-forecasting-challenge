import pandas as pd
import numpy as np


def flatten_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join([str(i) for i in col]) for col in df.columns]
    return df


def interpolate_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df.interpolate(method="time")
    df = df.ffill()
    return df


def aggregate_weather(df: pd.DataFrame) -> pd.DataFrame:
    ssrd_cols = [c for c in df.columns if c.endswith("ssrd")]
    tcc_cols  = [c for c in df.columns if c.endswith("tcc")]
    temp_cols = [c for c in df.columns if c.endswith("2t")]
    wind_cols = [c for c in df.columns if c.endswith("100ws")]

    flat = pd.DataFrame(index=df.index)
    flat["ssrd_mean"] = df[ssrd_cols].mean(axis=1)
    flat["ssrd_std"]  = df[ssrd_cols].std(axis=1)
    flat["tcc_mean"]  = df[tcc_cols].mean(axis=1)
    flat["tcc_std"]   = df[tcc_cols].std(axis=1)
    flat["temp_mean"] = df[temp_cols].mean(axis=1)
    flat["wind_mean"] = df[wind_cols].mean(axis=1)
    flat["wind_std"]  = df[wind_cols].std(axis=1)

    return flat.bfill().ffill()


def prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten multi-index columns, interpolate missing values, and aggregate to per-hour stats."""
    df = flatten_weather_columns(df)
    df = interpolate_weather(df)
    df = aggregate_weather(df)
    df = df.reset_index().rename(columns={df.index.name or "index": "ds"})
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ssrd_lag1"]  = df["ssrd_mean"].shift(1)
    df["ssrd_lag24"] = df["ssrd_mean"].shift(24)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ssrd_roll_3h"]     = df["ssrd_mean"].rolling(3).mean()
    df["ssrd_roll_24h"]    = df["ssrd_mean"].rolling(24).mean()
    df["tcc_roll_6h_std"]  = df["tcc_mean"].rolling(6).std()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["ds"])
    df["hour"]      = dt.dt.hour
    df["month"]     = dt.dt.month
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ssrd_norm"]       = df["ssrd_mean"] / df["ssrd_mean"].max()
    df["solar_potential"] = df["ssrd_norm"] * (1 - df["tcc_mean"])
    return df


def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lags, rolling stats, time encodings, and solar features, then drop NaN rows."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_solar_features(df)
    return df.dropna()
