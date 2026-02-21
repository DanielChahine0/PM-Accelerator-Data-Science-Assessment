"""
prophet_model.py – Facebook Prophet forecasting wrapper.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

# Suppress Prophet's verbose logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def fit_prophet(
    train: pd.DataFrame,
    horizon: int = 14,
    date_col: str = "date",
    target_col: str = "temperature_celsius",
) -> np.ndarray:
    """
    Fit Prophet and return *horizon*-step-ahead forecast.

    Parameters
    ----------
    train : DataFrame with columns [date_col, target_col]
    horizon : number of days to forecast
    """
    from prophet import Prophet  # lazy import to keep startup fast

    prophet_df = train[[date_col, target_col]].rename(
        columns={date_col: "ds", target_col: "y"}
    )
    prophet_df = prophet_df.dropna(subset=["ds", "y"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    return forecast["yhat"].iloc[-horizon:].values
