"""
tests/test_forecaster.py

Unit tests for multi-quantile forecaster and supply chain evaluation metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.generator import generate_hierarchical_demand
from src.data.preprocessor import HierarchicalPreprocessor, split_chronological
from src.forecasting.forecaster import MultiQuantileForecaster
from src.forecasting.metrics import pinball_loss, rmse, smape, wape


@pytest.fixture(scope="module")
def prepared_splits():
    raw = generate_hierarchical_demand(n_days=180, random_state=42)
    prep = HierarchicalPreprocessor()
    feat_df = prep.build_features(raw)
    return split_chronological(feat_df, test_days=14, val_days=14)


def test_metrics_calculation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])

    assert 0.0 <= wape(y_true, y_pred) <= 1.0
    assert 0.0 <= smape(y_true, y_pred) <= 100.0
    assert rmse(y_true, y_pred) > 0.0
    assert pinball_loss(y_true, y_pred, 0.5) > 0.0


def test_forecaster_quantiles(prepared_splits):
    train, val, test = prepared_splits
    forecaster = MultiQuantileForecaster(quantiles=[0.10, 0.50, 0.90])
    forecaster.fit(train, train["sales"].values, val, val["sales"].values)

    preds = forecaster.predict_quantiles(test)
    assert "pred_q10" in preds.columns
    assert "pred_q50" in preds.columns
    assert "pred_q90" in preds.columns

    # Verify quantile monotonicity: q10 <= q50 <= q90
    p10 = preds["pred_q10"].values
    p50 = preds["pred_q50"].values
    p90 = preds["pred_q90"].values

    assert (p10 <= p50 + 1e-5).all()
    assert (p50 <= p90 + 1e-5).all()
