"""
features.py – Feature engineering for ML regression forecasting.

Generates:
- Lag features (1, 2, 7, 14 days)
- Rolling statistics (7-day & 14-day mean/std)
- Calendar features (day of week, month, day of year, is_weekend)
- Cyclical encodings for month and day-of-week
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Lag features
# ──────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target: str = "temperature_celsius",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged values of *target* grouped by location."""
    if lags is None:
        lags = [1, 2, 7, 14]
    for lag in lags:
        df[f"{target}_lag{lag}"] = (
            df.groupby("location_name")[target].shift(lag)
        )
    return df


# ──────────────────────────────────────────────
# Rolling statistics
# ──────────────────────────────────────────────

def add_rolling_stats(
    df: pd.DataFrame,
    target: str = "temperature_celsius",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std for *target* grouped by location."""
    if windows is None:
        windows = [7, 14]
    for w in windows:
        grp = df.groupby("location_name")[target]
        df[f"{target}_roll_mean{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target}_roll_std{w}"] = grp.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
    return df


# ──────────────────────────────────────────────
# Calendar features
# ──────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract calendar-based features from the *date* column."""
    dt = df["date"].dt
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["day_of_year"] = dt.dayofyear
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


# ──────────────────────────────────────────────
# Master builder
# ──────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    target: str = "temperature_celsius",
) -> pd.DataFrame:
    """Run the full feature-engineering pipeline (returns a copy)."""
    df = df.copy()
    df = df.sort_values(["location_name", "date"]).reset_index(drop=True)
    df = add_lag_features(df, target)
    df = add_rolling_stats(df, target)
    df = add_calendar_features(df)
    return df
