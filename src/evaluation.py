"""
evaluation.py – Forecasting evaluation metrics.

Metrics: MAE, RMSE, sMAPE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale)."""
    denom = (np.abs(y_true) + np.abs(y_pred))
    # Avoid division by zero
    mask = denom > 0
    return float(
        200.0 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask])
    )


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "",
) -> dict:
    """Return a dict of evaluation metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "model": model_name,
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
    }
