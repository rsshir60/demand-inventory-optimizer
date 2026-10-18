"""
src/api/main.py

FastAPI application for Hierarchical Demand Forecasting & Linear Programming Inventory Optimization.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    DailyQuantilePrediction,
    ForecastRequest,
    ForecastResponse,
    RecommendedOrderItem,
    ReorderOptimizeRequest,
    ReorderOptimizeResponse,
    SimulationComparisonResponse,
)
from src.data.generator import ITEM_CATALOG, generate_hierarchical_demand
from src.data.preprocessor import HierarchicalPreprocessor, split_chronological
from src.forecasting.forecaster import MultiQuantileForecaster
from src.optimization.lp_solver import InventoryLPSolver, ItemOrderContext
from src.optimization.safety_stock import compute_reorder_point
from src.simulation.inventory_sim import InventorySimulator

ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "models"

APP_VERSION = "1.0.0"

_forecaster: MultiQuantileForecaster | None = None
_preprocessor: HierarchicalPreprocessor | None = None
_cached_test_df: pd.DataFrame | None = None
_cached_forecasts: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _forecaster, _preprocessor, _cached_test_df, _cached_forecasts
    logger.info("Initializing Demand Forecasting & Inventory Optimization service ...")

    model_path = ARTIFACT_DIR / "quantile_forecaster.joblib"
    if model_path.exists():
        _forecaster = MultiQuantileForecaster.load(model_path)
    else:
        logger.warning(f"No trained forecaster found at {model_path}. Training initial baseline ...")
        raw_df = generate_hierarchical_demand(n_days=730)
        _preprocessor = HierarchicalPreprocessor()
        feat_df = _preprocessor.build_features(raw_df)
        train, val, test = split_chronological(feat_df)
        _forecaster = MultiQuantileForecaster()
        _forecaster.fit(train, train["sales"].values, val, val["sales"].values)
        _forecaster.save(model_path)
        _cached_test_df = test

    # Generate cached test forecasts for fast simulation runs
    if _cached_test_df is None:
        raw_df = generate_hierarchical_demand(n_days=730)
        if _preprocessor is None:
            _preprocessor = HierarchicalPreprocessor()
        feat_df = _preprocessor.build_features(raw_df)
        _, _, _cached_test_df = split_chronological(feat_df)

    preds = _forecaster.predict_quantiles(_cached_test_df)
    _cached_forecasts = pd.concat([_cached_test_df[["date", "store_id", "item_id"]], preds], axis=1)

    logger.info("Service initialized successfully.")
    yield
    logger.info("Service shutting down.")


app = FastAPI(
    title="Hierarchical Demand Forecasting & Inventory LP Solver",
    description="Multi-quantile LightGBM demand predictions paired with HiGHS Linear Programming inventory optimization",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["ops"])
async def health():
    return {
        "status": "ok",
        "model_loaded": _forecaster is not None,
        "items_catalog_size": len(ITEM_CATALOG),
        "version": APP_VERSION,
    }


@app.post("/forecast/quantiles", response_model=ForecastResponse, tags=["forecasting"])
async def predict_forecast(req: ForecastRequest):
    """
    Returns multi-horizon quantile forecast (P10, P50, P90) and dynamic safety stock.
    """
    if req.item_id not in ITEM_CATALOG:
        raise HTTPException(status_code=404, detail=f"Item '{req.item_id}' not found in catalog")

    meta = ITEM_CATALOG[req.item_id]
    # Filter cached predictions for this series
    slice_df = _cached_forecasts[
        (_cached_forecasts["store_id"] == req.store_id) & (_cached_forecasts["item_id"] == req.item_id)
    ].head(req.horizon_days)

    daily_preds = []
    p50_vals, p90_vals = [], []
    for idx, (_, row) in enumerate(slice_df.iterrows(), start=1):
        p10 = round(float(row["pred_q10"]), 1)
        p50 = round(float(row["pred_q50"]), 1)
        p90 = round(float(row["pred_q90"]), 1)
        p50_vals.append(p50)
        p90_vals.append(p90)
        daily_preds.append(DailyQuantilePrediction(day_offset=idx, p10=p10, p50=p50, p90=p90))

    mean_p50 = float(sum(p50_vals) / len(p50_vals)) if p50_vals else 20.0
    mean_p90 = float(sum(p90_vals) / len(p90_vals)) if p90_vals else 30.0

    rop_info = compute_reorder_point(
        pred_p50_daily=mean_p50,
        pred_p90_daily=mean_p90,
        lead_time_days=meta.lead_time_days,
    )

    return ForecastResponse(
        store_id=req.store_id,
        item_id=req.item_id,
        horizon_days=len(daily_preds),
        daily_predictions=daily_preds,
        lead_time_days=meta.lead_time_days,
        dynamic_safety_stock=rop_info["safety_stock"],
        reorder_point=rop_info["reorder_point"],
    )


@app.post("/inventory/optimize", response_model=ReorderOptimizeResponse, tags=["optimization"])
async def optimize_reorder(req: ReorderOptimizeRequest):
    """
    Solves the Constrained Reorder Linear Program across stores and items under budget and capacity limits.
    """
    solver = InventoryLPSolver(
        total_procurement_budget=req.total_budget,
        store_volume_capacities=req.store_capacities,
    )

    items: List[ItemOrderContext] = []
    for inv in req.inventory_levels:
        if inv.item_id not in ITEM_CATALOG:
            continue
        meta = ITEM_CATALOG[inv.item_id]

        # Get predicted demand
        slice_df = _cached_forecasts[
            (_cached_forecasts["store_id"] == inv.store_id) & (_cached_forecasts["item_id"] == inv.item_id)
        ]
        p50 = float(slice_df["pred_q50"].mean()) if not slice_df.empty else 20.0
        p90 = float(slice_df["pred_q90"].mean()) if not slice_df.empty else 30.0

        rop_info = compute_reorder_point(p50, p90, meta.lead_time_days)

        items.append(
            ItemOrderContext(
                store_id=inv.store_id,
                item_id=inv.item_id,
                current_inventory=inv.current_inventory,
                target_rop=rop_info["reorder_point"],
                unit_cost=meta.unit_cost,
                holding_cost_daily=meta.holding_cost_daily,
                stockout_penalty=meta.stockout_penalty,
                unit_volume_m3=meta.unit_volume_m3,
            )
        )

    res = solver.solve(items)

    recommendations = []
    for it in items:
        q_val = res.orders.get((it.store_id, it.item_id), 0.0)
        recommendations.append(
            RecommendedOrderItem(
                store_id=it.store_id,
                item_id=it.item_id,
                current_inventory=it.current_inventory,
                target_rop=it.target_rop,
                recommended_order_units=q_val,
                unit_cost=it.unit_cost,
                subtotal_cost=round(q_val * it.unit_cost, 2),
            )
        )

    return ReorderOptimizeResponse(
        success=res.success,
        status_message=res.status_message,
        total_procurement_cost=res.total_procurement_cost,
        budget_limit=res.budget_limit,
        budget_utilization_pct=res.budget_utilization_pct,
        store_volume_utilization=res.store_volume_utilization,
        recommendations=recommendations,
    )


@app.post("/inventory/simulate", response_model=SimulationComparisonResponse, tags=["simulation"])
async def run_simulation_benchmark():
    """
    Executes 30-day inventory simulation comparing LP Quantile Optimization vs Moving Average baseline.
    """
    solver = InventoryLPSolver()
    sim = InventorySimulator(
        test_df=_cached_test_df,
        forecast_df=_cached_forecasts,
        solver=solver,
    )
    results = sim.run_simulation()
    opt = results["LP_Quantile_Optimal"]
    base = results["Baseline_Moving_Average"]

    diff_stockout = base.stockout_rate_pct - opt.stockout_rate_pct
    stockout_red = (diff_stockout / base.stockout_rate_pct) * 100.0 if base.stockout_rate_pct > 0 else 0.0
    savings = base.total_operating_cost - opt.total_operating_cost
    cost_red = (savings / base.total_operating_cost) * 100.0 if base.total_operating_cost > 0 else 0.0

    return SimulationComparisonResponse(
        baseline_stockout_rate_pct=base.stockout_rate_pct,
        optimal_stockout_rate_pct=opt.stockout_rate_pct,
        stockout_reduction_pct=round(stockout_red, 1),
        baseline_operating_cost=base.total_operating_cost,
        optimal_operating_cost=opt.total_operating_cost,
        net_dollar_savings=round(savings, 2),
        operating_cost_reduction_pct=round(cost_red, 1),
    )
