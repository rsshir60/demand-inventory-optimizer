"""
tests/test_data.py

Unit tests for hierarchical data generation and lag preprocessing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.generator import ITEM_CATALOG, STORES, generate_hierarchical_demand
from src.data.preprocessor import HierarchicalPreprocessor, split_chronological, LAG_FEATURES


@pytest.fixture(scope="module")
def raw_df():
    return generate_hierarchical_demand(n_days=180, random_state=0)


def test_generator_structure(raw_df):
    assert len(raw_df) > 0
    assert set(raw_df["store_id"].unique()) == set(STORES)
    assert set(raw_df["item_id"].unique()) == set(ITEM_CATALOG.keys())
    assert (raw_df["sales"] >= 0).all()


def test_preprocessor_lags(raw_df):
    prep = HierarchicalPreprocessor()
    feat_df = prep.build_features(raw_df)

    assert not feat_df.empty
    for lag_col in LAG_FEATURES:
        assert lag_col in feat_df.columns
        assert not feat_df[lag_col].isna().any()


def test_chronological_split(raw_df):
    prep = HierarchicalPreprocessor()
    feat_df = prep.build_features(raw_df)
    train, val, test = split_chronological(feat_df, test_days=14, val_days=14)

    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(feat_df)
