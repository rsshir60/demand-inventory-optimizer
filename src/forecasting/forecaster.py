"""
src/forecasting/forecaster.py

Multi-Quantile LightGBM Forecaster for demand distribution estimation.
Predicts P10 (optimistic holding cost), P50 (expected demand), and P90 (surge risk buffer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from src.data.preprocessor import CATEGORICAL_FEATURES, MODEL_FEATURE_COLS

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "models"

DEFAULT_PARAMS = {
    "boosting_type": "gbdt",
    "n_estimators": 350,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


class MultiQuantileForecaster:
    """Trains 3 LightGBM quantile regression models: q=0.10, q=0.50, q=0.90."""

    def __init__(self, quantiles: List[float] | None = None):
        self.quantiles = quantiles or [0.10, 0.50, 0.90]
        self.models: Dict[float, lgb.LGBMRegressor] = {}
        self.feature_names = MODEL_FEATURE_COLS.copy()

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "MultiQuantileForecaster":
        X_tr = X_train[self.feature_names]
        X_v = X_val[self.feature_names]

        for q in self.quantiles:
            logger.info(f"Training LightGBM Quantile Regressor for q={q:.2f} ...")
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **DEFAULT_PARAMS,
            )
            model.fit(
                X_tr,
                y_train,
                eval_set=[(X_v, y_val)],
                categorical_feature=CATEGORICAL_FEATURES,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            self.models[q] = model
            logger.info(f"  q={q:.2f} converged at iteration {model.best_iteration_}")

        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.models:
            raise RuntimeError("Models not fitted.")
        X_in = X[self.feature_names]
        preds = {}
        for q, model in self.models.items():
            col_name = f"pred_q{int(q * 100)}"
            # Demands cannot be negative
            preds[col_name] = np.maximum(0.0, model.predict(X_in))

        # Enforce quantile monotonicity: q10 <= q50 <= q90
        preds_df = pd.DataFrame(preds, index=X.index)
        if "pred_q10" in preds_df and "pred_q50" in preds_df:
            preds_df["pred_q50"] = np.maximum(preds_df["pred_q50"], preds_df["pred_q10"])
        if "pred_q50" in preds_df and "pred_q90" in preds_df:
            preds_df["pred_q90"] = np.maximum(preds_df["pred_q90"], preds_df["pred_q50"])

        return preds_df

    def feature_importances(self) -> pd.DataFrame:
        if 0.50 not in self.models:
            return pd.DataFrame()
        imp = self.models[0.50].feature_importances_
        return pd.DataFrame({"feature": self.feature_names, "importance": imp}).sort_values("importance", ascending=False)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = path or ARTIFACT_DIR / "quantile_forecaster.joblib"
        joblib.dump(self, path)
        logger.info(f"Forecaster saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "MultiQuantileForecaster":
        path = path or ARTIFACT_DIR / "quantile_forecaster.joblib"
        obj = joblib.load(path)
        logger.info(f"Forecaster loaded from {path}")
        return obj
