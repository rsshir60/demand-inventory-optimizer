"""
src/optimization/safety_stock.py

Translates quantile demand uncertainty into dynamic safety stock and Reorder Points (ROP).

Rather than assuming constant normal variance (sigma), this module leverages the
LightGBM P90 - P50 prediction spread to adaptively inflate safety buffers
ahead of weekends, holidays, and high-elasticity promotional events.
"""

from __future__ import annotations

import numpy as np


def compute_dynamic_safety_stock(
    pred_p50_daily: float,
    pred_p90_daily: float,
    lead_time_days: int,
    service_level_z: float = 1.65,  # ~95% cycle service level
    min_buffer_units: float = 2.0,
) -> float:
    """
    Calculates safety stock buffer (units) considering lead time.
    Spread = (P90 - P50) represents empirical right-tail demand uncertainty.
    """
    # Daily empirical standard deviation surrogate from 90th percentile:
    # For normal distribution, (P90 - P50) approx 1.28 * sigma => sigma approx spread / 1.28
    spread = max(0.1, pred_p90_daily - pred_p50_daily)
    sigma_daily = spread / 1.2816

    # Lead time demand standard deviation: sigma_L = sqrt(L) * sigma_D
    sigma_lead_time = np.sqrt(lead_time_days) * sigma_daily
    safety_stock = max(min_buffer_units, service_level_z * sigma_lead_time)
    return float(np.round(safety_stock, 1))


def compute_reorder_point(
    pred_p50_daily: float,
    pred_p90_daily: float,
    lead_time_days: int,
    service_level_z: float = 1.65,
) -> dict[str, float]:
    """
    Reorder Point (ROP) = Expected Lead Time Demand + Safety Stock
    """
    lead_time_demand = pred_p50_daily * lead_time_days
    ss = compute_dynamic_safety_stock(
        pred_p50_daily=pred_p50_daily,
        pred_p90_daily=pred_p90_daily,
        lead_time_days=lead_time_days,
        service_level_z=service_level_z,
    )
    rop = lead_time_demand + ss
    return {
        "lead_time_demand": round(lead_time_demand, 1),
        "safety_stock": ss,
        "reorder_point": round(rop, 1),
    }
