"""
ensemble.py – Weighted ensemble of forecasting models.

Combines individual model predictions using inverse-MAE weighting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.evaluation import mae


def inverse_mae_weights(
    y_val: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute normalised weights as 1/MAE for each model.

    Parameters
    ----------
    y_val : array of validation actuals
    predictions : {model_name: prediction_array}

    Returns
    -------
    dict of model_name → weight  (sums to 1)
    """
    inv_maes = {}
    for name, preds in predictions.items():
        m = mae(y_val, preds)
        inv_maes[name] = 1.0 / max(m, 1e-8)
    total = sum(inv_maes.values())
    return {k: v / total for k, v in inv_maes.items()}


def weighted_ensemble(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Return weighted average of model predictions.
    """
    result = np.zeros_like(list(predictions.values())[0], dtype=float)
    for name, preds in predictions.items():
        w = weights.get(name, 0.0)
        result += w * preds
    return result
