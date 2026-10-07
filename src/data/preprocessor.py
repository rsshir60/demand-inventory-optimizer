"""
src/data/preprocessor.py

Feature engineering and chronological split pipeline for hierarchical time series.
Enforces strict multi-horizon lag safety: lag >= 7 days prevents target leakage
when generating weekly replenishment forecasts.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import OrdinalEncoder

CALENDAR_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_promo",
]

LAG_FEATURES = [
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_21",
    "sales_lag_28",
    "sales_roll_mean_7",
    "sales_roll_std_7",
    "sales_roll_mean_28",
]

CATEGORICAL_FEATURES = ["store_encoded", "item_encoded"]

MODEL_FEATURE_COLS = CALENDAR_FEATURES + LAG_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "sales"


class HierarchicalPreprocessor:
    """Creates lag, rolling window, calendar, and hierarchical features."""

    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.is_fitted = False

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

        # 1. Calendar Features
        out["day_of_week"] = out["date"].dt.dayofweek
        out["day_of_month"] = out["date"].dt.day
        out["month"] = out["date"].dt.month
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

        # 2. Store & Item Categorical Encoding
        if not self.is_fitted:
            encoded = self.encoder.fit_transform(out[["store_id", "item_id"]])
            self.is_fitted = True
        else:
            encoded = self.encoder.transform(out[["store_id", "item_id"]])

        out["store_encoded"] = encoded[:, 0].astype(int)
        out["item_encoded"] = encoded[:, 1].astype(int)

        # 3. Grouped Lags & Rolling Statistics (per store-item series)
        # Shift >= 7 ensures that predictions 1-7 days ahead don't leak unobserved values
        grouped = out.groupby(["store_id", "item_id"])["sales"]

        out["sales_lag_7"] = grouped.shift(7)
        out["sales_lag_14"] = grouped.shift(14)
        out["sales_lag_21"] = grouped.shift(21)
        out["sales_lag_28"] = grouped.shift(28)

        # Rolling statistics on shifted series
        out["sales_roll_mean_7"] = grouped.transform(lambda s: s.shift(7).rolling(7, min_periods=3).mean())
        out["sales_roll_std_7"] = grouped.transform(lambda s: s.shift(7).rolling(7, min_periods=3).std().fillna(0))
        out["sales_roll_mean_28"] = grouped.transform(lambda s: s.shift(7).rolling(28, min_periods=7).mean())

        # Drop initial warm-up rows containing NaNs
        clean_df = out.dropna(subset=LAG_FEATURES).reset_index(drop=True)
        return clean_df


def split_chronological(
    df: pd.DataFrame,
    test_days: int = 30,
    val_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological out-of-time split. Never shuffles across time.
    """
    max_date = df["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train = df[df["date"] < val_cutoff].reset_index(drop=True)
    val = df[(df["date"] >= val_cutoff) & (df["date"] < test_cutoff)].reset_index(drop=True)
    test = df[df["date"] >= test_cutoff].reset_index(drop=True)

    logger.info(
        f"Chronological split:\n"
        f"  Train : {train['date'].min().date()} to {train['date'].max().date()} ({len(train):,} rows)\n"
        f"  Val   : {val['date'].min().date()} to {val['date'].max().date()} ({len(val):,} rows)\n"
        f"  Test  : {test['date'].min().date()} to {test['date'].max().date()} ({len(test):,} rows)"
    )
    return train, val, test
