"""
src/forecasting/metrics.py

Supply chain and retail forecasting evaluation metrics:
- WAPE (Weighted Absolute Percentage Error): sum(|y - y_hat|) / sum(y)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- Quantile / Pinball Loss for probabilistic uncertainty evaluation
- RMSE
"""

from __future__ import annotations

import numpy as np


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    The gold standard in supply chain; robust against intermittent/zero-demand days.
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error in percentage [0, 100]."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Pinball / Quantile loss:
        L_q(y, y_hat) = max(q * (y - y_hat), (1 - q) * (y_hat - y))
    """
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))
