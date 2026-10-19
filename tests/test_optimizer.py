"""
tests/test_optimizer.py

Unit tests for dynamic safety stock buffers and Linear Programming inventory optimization.
"""

from __future__ import annotations

import pytest
from src.optimization.lp_solver import InventoryLPSolver, ItemOrderContext
from src.optimization.safety_stock import compute_dynamic_safety_stock, compute_reorder_point


def test_dynamic_safety_stock():
    # Wider spread (high uncertainty) must produce a larger safety stock
    ss_low_var = compute_dynamic_safety_stock(pred_p50_daily=50.0, pred_p90_daily=55.0, lead_time_days=3)
    ss_high_var = compute_dynamic_safety_stock(pred_p50_daily=50.0, pred_p90_daily=85.0, lead_time_days=3)

    assert ss_high_var > ss_low_var
    assert ss_low_var >= 2.0


def test_reorder_point():
    res = compute_reorder_point(pred_p50_daily=20.0, pred_p90_daily=30.0, lead_time_days=3)
    assert "lead_time_demand" in res
    assert "safety_stock" in res
    assert "reorder_point" in res
    assert res["reorder_point"] == res["lead_time_demand"] + res["safety_stock"]


def test_lp_solver_budget_constraint():
    solver = InventoryLPSolver(total_procurement_budget=1000.0)

    # 2 items needing heavy replenishment
    items = [
        ItemOrderContext(
            store_id="STORE_A",
            item_id="ITEM_01",
            current_inventory=5.0,
            target_rop=100.0,
            unit_cost=10.0,
            holding_cost_daily=0.1,
            stockout_penalty=20.0,
            unit_volume_m3=0.01,
        ),
        ItemOrderContext(
            store_id="STORE_A",
            item_id="ITEM_02",
            current_inventory=0.0,
            target_rop=80.0,
            unit_cost=50.0,
            holding_cost_daily=0.5,
            stockout_penalty=100.0,
            unit_volume_m3=0.04,
        ),
    ]

    res = solver.solve(items)
    assert res.success is True
    # Budget strictly respected:
    assert res.budget_used <= 1000.0 + 1e-3
    assert res.budget_utilization_pct <= 100.0
