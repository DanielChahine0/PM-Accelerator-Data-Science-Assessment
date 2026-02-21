"""
anomalies.py – Anomaly detection via STL decomposition & Isolation Forest.

• STL residual-based anomaly flagging (|residual| > 3σ)
• Isolation Forest on multivariate weather features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL


# ──────────────────────────────────────────────
# STL anomaly detection
# ──────────────────────────────────────────────

def stl_anomaly_detection(
    series: pd.Series,
    period: int = 7,
    threshold_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Run STL decomposition on a time series and flag residuals
    exceeding *threshold_sigma* standard deviations.

    Returns DataFrame with columns: trend, seasonal, resid, anomaly.
    """
    series = series.dropna()
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    resid = result.resid
    sigma = resid.std()
    anomaly = (resid.abs() > threshold_sigma * sigma)
    return pd.DataFrame({
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": resid,
        "anomaly": anomaly,
    })


# ──────────────────────────────────────────────
# Isolation Forest
# ──────────────────────────────────────────────

def isolation_forest_anomalies(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.Series:
    """
    Fit Isolation Forest and return a boolean Series of anomalies.
    """
    if feature_cols is None:
        feature_cols = [
            "temperature_celsius",
            "humidity",
            "precip_mm",
            "wind_kph",
            "pressure_mb",
        ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].dropna()
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X)
    # -1 → anomaly
    result = pd.Series(False, index=df.index, name="iso_forest_anomaly")
    result.loc[X.index] = preds == -1
    return result
