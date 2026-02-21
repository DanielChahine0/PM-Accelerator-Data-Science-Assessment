"""
ml_regression.py – Gradient Boosting regression forecaster.

Uses lag features, rolling statistics, and calendar features built by features.py.
Returns predictions, trained model, and feature importances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


# Default feature columns (must already exist in the df)
FEATURE_COLS = [
    "temperature_celsius_lag1",
    "temperature_celsius_lag2",
    "temperature_celsius_lag7",
    "temperature_celsius_lag14",
    "temperature_celsius_roll_mean7",
    "temperature_celsius_roll_std7",
    "temperature_celsius_roll_mean14",
    "temperature_celsius_roll_std14",
    "humidity",
    "wind_kph",
    "pressure_mb",
    "cloud",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "latitude",
    "longitude",
]

TARGET = "temperature_celsius"


def _prepare(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop rows where any feature or target is NaN."""
    cols = feature_cols + [TARGET]
    available = [c for c in cols if c in df.columns]
    return df.dropna(subset=available)


def train_ml_model(
    train_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    **kwargs,
) -> tuple[GradientBoostingRegressor, list[str]]:
    """
    Train a GradientBoostingRegressor and return (model, used_feature_cols).
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    df = _prepare(train_df, feature_cols)
    X = df[feature_cols]
    y = df[TARGET]

    params = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    params.update(kwargs)
    model = GradientBoostingRegressor(**params)
    model.fit(X, y)
    return model, feature_cols


def predict_ml(
    model: GradientBoostingRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """Generate predictions for *df* using *model*."""
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].fillna(0)
    return model.predict(X)


def feature_importance(
    model: GradientBoostingRegressor,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return sorted feature importance DataFrame."""
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return imp
