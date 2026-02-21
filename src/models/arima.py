"""
arima.py – SARIMA forecasting wrapper.

Uses statsmodels SARIMAX with configurable order and seasonal_order.
Falls back to simpler model if convergence fails.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def fit_sarima(
    train: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 7),
    horizon: int = 14,
) -> np.ndarray:
    """
    Fit SARIMA and return *horizon*-step forecast.

    Falls back to (1,1,1)x(0,0,0,0) if seasonal model fails.
    """
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        forecast = result.forecast(steps=horizon)
        return forecast.values
    except Exception:
        # Fallback to non-seasonal ARIMA
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            forecast = result.forecast(steps=horizon)
            return forecast.values
        except Exception:
            # Ultimate fallback: return last value
            return np.full(horizon, train.iloc[-1])
