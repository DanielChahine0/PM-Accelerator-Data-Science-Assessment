"""
baseline.py – Naive and Seasonal Naive forecasting baselines.

• Naive: forecast = last observed value
• Seasonal Naive: forecast = value from *season* days ago
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value for *horizon* steps."""
    last = train.iloc[-1]
    return np.full(horizon, last)


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    season: int = 7,
) -> np.ndarray:
    """
    Repeat the last *season*-length cycle for *horizon* steps.
    """
    tail = train.iloc[-season:].values
    reps = int(np.ceil(horizon / season))
    return np.tile(tail, reps)[:horizon]
