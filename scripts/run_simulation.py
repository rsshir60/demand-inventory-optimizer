"""
scripts/run_simulation.py

Runs 30-day discrete-event simulation comparing:
1. LP Quantile Optimizer (Our Solution)
2. Moving Average Baseline Policy

Evaluates Stockout Rate, Service Level, Holding Costs, Stockout Penalties, and Net ROI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import typer
from loguru import logger
from tabulate import tabulate

from src.data.generator import generate_hierarchical_demand
from src.data.preprocessor import HierarchicalPreprocessor, split_chronological
from src.forecasting.forecaster import MultiQuantileForecaster
from src.optimization.lp_solver import InventoryLPSolver
from src.simulation.inventory_sim import InventorySimulator

REPORT_DIR = ROOT / "artifacts" / "simulation_reports"
MODEL_PATH = ROOT / "artifacts" / "models" / "quantile_forecaster.joblib"

app = typer.Typer(add_completion=False)


@app.command()
def simulate():
    logger.info("Setting up simulation environment ...")
    raw_df = generate_hierarchical_demand(n_days=730)
    preprocessor = HierarchicalPreprocessor()
    feat_df = preprocessor.build_features(raw_df)
    train_df, val_df, test_df = split_chronological(feat_df, test_days=30, val_days=30)

    if MODEL_PATH.exists():
        forecaster = MultiQuantileForecaster.load(MODEL_PATH)
    else:
        logger.info("Training forecaster ...")
        forecaster = MultiQuantileForecaster()
        forecaster.fit(train_df, train_df["sales"].values, val_df, val_df["sales"].values)
        forecaster.save(MODEL_PATH)

    preds = forecaster.predict_quantiles(test_df)
    forecast_df = pd.concat([test_df[["date", "store_id", "item_id"]], preds], axis=1)

    solver = InventoryLPSolver(total_procurement_budget=40_000.0)
    sim = InventorySimulator(test_df=test_df, forecast_df=forecast_df, solver=solver)

    results = sim.run_simulation()
    opt = results["LP_Quantile_Optimal"]
    base = results["Baseline_Moving_Average"]

    savings = base.total_operating_cost - opt.total_operating_cost
    pct_savings = (savings / base.total_operating_cost) * 100 if base.total_operating_cost > 0 else 0

    table_data = [
        ["Total Demand (Units)", f"{base.total_demand_units:,.0f}", f"{opt.total_demand_units:,.0f}", "—"],
        ["Fulfilled Demand (Units)", f"{base.total_fulfilled_units:,.0f}", f"{opt.total_fulfilled_units:,.0f}", f"+{opt.total_fulfilled_units - base.total_fulfilled_units:,.0f}"],
        ["Lost Sales (Units)", f"{base.total_lost_sales_units:,.0f}", f"{opt.total_lost_sales_units:,.0f}", f"-{base.total_lost_sales_units - opt.total_lost_sales_units:,.0f}"],
        ["Stockout Rate (%)", f"{base.stockout_rate_pct:.1f}%", f"{opt.stockout_rate_pct:.1f}%", f"-{base.stockout_rate_pct - opt.stockout_rate_pct:.1f}%"],
        ["Service Level (%)", f"{base.service_level_pct:.1f}%", f"{opt.service_level_pct:.1f}%", f"+{opt.service_level_pct - base.service_level_pct:.1f}%"],
        ["Holding Cost ($)", f"${base.total_holding_cost:,.2f}", f"${opt.total_holding_cost:,.2f}", f"-${base.total_holding_cost - opt.total_holding_cost:,.2f}"],
        ["Stockout Penalty ($)", f"${base.total_stockout_penalty:,.2f}", f"${opt.total_stockout_penalty:,.2f}", f"-${base.total_stockout_penalty - opt.total_stockout_penalty:,.2f}"],
        ["Total Operating Cost ($)", f"${base.total_operating_cost:,.2f}", f"${opt.total_operating_cost:,.2f}", f"-${savings:,.2f} ({pct_savings:.1f}%)"],
    ]

    print("\n" + "=" * 80)
    print("30-DAY INVENTORY SIMULATION: BASELINE VS LP QUANTILE OPTIMIZER")
    print("=" * 80)
    print(tabulate(table_data, headers=["Operational Metric", "Moving Average Baseline", "LP Quantile Optimal", "Net Improvement"], tablefmt="github"))
    print("=" * 80 + "\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORT_DIR / "simulation_summary.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Inventory Optimization Simulation Benchmark\n\n")
        f.write(tabulate(table_data, headers=["Operational Metric", "Moving Average Baseline", "LP Quantile Optimal", "Net Improvement"], tablefmt="github"))
        f.write("\n")
    logger.info(f"Saved simulation report to {out_file}")


if __name__ == "__main__":
    app()
