"""
scripts/train_forecast.py

End-to-end training script for Multi-Quantile Hierarchical Demand Forecasting.
Evaluates WAPE, sMAPE, RMSE, and Pinball Loss on out-of-time test split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import typer
from loguru import logger
from tabulate import tabulate

from src.data.generator import generate_hierarchical_demand
from src.data.preprocessor import HierarchicalPreprocessor, split_chronological
from src.forecasting.forecaster import MultiQuantileForecaster
from src.forecasting.metrics import pinball_loss, rmse, smape, wape

app = typer.Typer(add_completion=False)


@app.command()
def train(
    n_days: int = typer.Option(730, help="Days of history to simulate"),
    random_state: int = typer.Option(42),
):
    logger.info("Generating hierarchical sales history ...")
    raw_df = generate_hierarchical_demand(n_days=n_days, random_state=random_state)

    logger.info("Building multi-horizon lag and rolling features ...")
    preprocessor = HierarchicalPreprocessor()
    feat_df = preprocessor.build_features(raw_df)

    train_df, val_df, test_df = split_chronological(feat_df, test_days=30, val_days=30)

    y_train = train_df["sales"].values
    y_val = val_df["sales"].values
    y_test = test_df["sales"].values

    forecaster = MultiQuantileForecaster(quantiles=[0.10, 0.50, 0.90])
    forecaster.fit(train_df, y_train, val_df, y_val)

    # Evaluate on out-of-time Test set
    preds = forecaster.predict_quantiles(test_df)

    p50 = preds["pred_q50"].values
    p10 = preds["pred_q10"].values
    p90 = preds["pred_q90"].values

    wape_val = wape(y_test, p50)
    smape_val = smape(y_test, p50)
    rmse_val = rmse(y_test, p50)
    pb_10 = pinball_loss(y_test, p10, 0.10)
    pb_50 = pinball_loss(y_test, p50, 0.50)
    pb_90 = pinball_loss(y_test, p90, 0.90)

    metrics_table = [
        ["WAPE (P50)", f"{wape_val * 100:.2f}%"],
        ["sMAPE (P50)", f"{smape_val:.2f}%"],
        ["RMSE", f"{rmse_val:.2f}"],
        ["Pinball Loss (P10)", f"{pb_10:.3f}"],
        ["Pinball Loss (P50)", f"{pb_50:.3f}"],
        ["Pinball Loss (P90)", f"{pb_90:.3f}"],
    ]

    print("\n" + "=" * 50)
    print("DEMAND FORECASTING EVALUATION (OUT-OF-TIME TEST)")
    print("=" * 50)
    print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="github"))
    print("=" * 50 + "\n")

    # Save artifact
    forecaster.save()
    logger.info("Forecaster saved successfully.")


if __name__ == "__main__":
    app()
