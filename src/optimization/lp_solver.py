"""
src/optimization/lp_solver.py

Constrained Multi-Item, Multi-Store Inventory Reorder Optimizer using Linear Programming (SciPy HiGHS).

Minimizes total system cost (Procurement + Holding + Unmet Stockout Penalty) subject to:
1. Reorder Target Constraint: On_Hand + Reorder_Qty + Shortage >= Target_ROP
2. Global Procurement Budget Constraint: sum(unit_cost * Reorder_Qty) <= Total_Budget
3. Warehouse Physical Volume Constraint (per store): sum(volume * (On_Hand + Reorder_Qty)) <= Max_Store_Volume
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from loguru import logger
from scipy.optimize import linprog


@dataclass
class ItemOrderContext:
    store_id: str
    item_id: str
    current_inventory: float
    target_rop: float
    unit_cost: float
    holding_cost_daily: float
    stockout_penalty: float
    unit_volume_m3: float


@dataclass
class LPReorderResult:
    success: bool
    status_message: str
    orders: Dict[Tuple[str, str], float]       # (store, item) -> order quantity
    shortages: Dict[Tuple[str, str], float]    # (store, item) -> unmet target shortage
    total_procurement_cost: float
    total_holding_cost: float
    total_penalty_cost: float
    total_projected_cost: float
    budget_limit: float
    budget_used: float
    budget_utilization_pct: float
    store_volume_utilization: Dict[str, float]


class InventoryLPSolver:
    """
    Linear Programming solver using SciPy HiGHS.
    """

    def __init__(
        self,
        total_procurement_budget: float = 35_000.0,
        store_volume_capacities: Dict[str, float] | None = None,
        planning_days: int = 7,
    ):
        self.total_budget = total_procurement_budget
        self.store_volume_capacities = store_volume_capacities or {
            "STORE_A": 45.0,  # cubic meters
            "STORE_B": 35.0,
            "STORE_C": 25.0,
        }
        self.planning_days = planning_days

    def solve(self, items: List[ItemOrderContext]) -> LPReorderResult:
        n = len(items)
        if n == 0:
            raise ValueError("No items provided to LP solver.")

        # Decision variables: [Q_0, ..., Q_{n-1}, S_0, ..., S_{n-1}]
        # Length = 2 * n
        # Q_i: Order Quantity >= 0
        # S_i: Shortage below ROP >= 0

        # Objective Function coefficients:
        # Cost = sum [ (unit_cost + holding_cost_daily * planning_days * 0.5) * Q_i + stockout_penalty * S_i ]
        c = np.zeros(2 * n)
        for i, it in enumerate(items):
            c[i] = it.unit_cost + (it.holding_cost_daily * self.planning_days * 0.5)
            c[n + i] = it.stockout_penalty

        # Constraints (A_ub * x <= b_ub)
        A_ub_list = []
        b_ub_list = []

        # 1. Target ROP fulfillment constraint:
        # Current_Inv + Q_i + S_i >= ROP  <=>  -Q_i - S_i <= Current_Inv - ROP
        for i, it in enumerate(items):
            row = np.zeros(2 * n)
            row[i] = -1.0
            row[n + i] = -1.0
            A_ub_list.append(row)
            b_ub_list.append(it.current_inventory - it.target_rop)

        # 2. Total Procurement Budget Constraint:
        # sum(unit_cost_i * Q_i) <= Total_Budget
        budget_row = np.zeros(2 * n)
        for i, it in enumerate(items):
            budget_row[i] = it.unit_cost
        A_ub_list.append(budget_row)
        b_ub_list.append(self.total_budget)

        # 3. Warehouse Volume Capacity Constraint per store:
        # sum_{i in store} unit_volume * (Current_Inv + Q_i) <= Store_Cap
        # sum_{i in store} unit_volume * Q_i <= Store_Cap - sum_{i in store} unit_volume * Current_Inv
        unique_stores = sorted(list(set(it.store_id for it in items)))
        for s_id in unique_stores:
            vol_row = np.zeros(2 * n)
            cap = self.store_volume_capacities.get(s_id, 50.0)
            current_vol = sum(it.unit_volume_m3 * it.current_inventory for it in items if it.store_id == s_id)
            for i, it in enumerate(items):
                if it.store_id == s_id:
                    vol_row[i] = it.unit_volume_m3
            A_ub_list.append(vol_row)
            b_ub_list.append(max(0.0, cap - current_vol))

        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)

        # Bounds: all Q_i >= 0, S_i >= 0
        bounds = [(0, None) for _ in range(2 * n)]

        logger.info(f"Solving Linear Program: {2 * n} variables, {len(b_ub)} constraints using HiGHS ...")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not res.success:
            logger.warning(f"LP optimization ended with status: {res.message}")

        orders: Dict[Tuple[str, str], float] = {}
        shortages: Dict[Tuple[str, str], float] = {}
        total_proc = 0.0
        total_hold = 0.0
        total_pen = 0.0

        for i, it in enumerate(items):
            q_val = float(np.round(res.x[i], 1)) if res.x is not None else 0.0
            s_val = float(np.round(res.x[n + i], 1)) if res.x is not None else 0.0
            orders[(it.store_id, it.item_id)] = q_val
            shortages[(it.store_id, it.item_id)] = s_val

            total_proc += q_val * it.unit_cost
            total_hold += (it.current_inventory + q_val * 0.5) * it.holding_cost_daily * self.planning_days
            total_pen += s_val * it.stockout_penalty

        # Volume utilization calculation
        vol_util: Dict[str, float] = {}
        for s_id in unique_stores:
            cap = self.store_volume_capacities.get(s_id, 50.0)
            used = sum(it.unit_volume_m3 * (it.current_inventory + orders.get((it.store_id, it.item_id), 0.0)) for it in items if it.store_id == s_id)
            vol_util[s_id] = round((used / cap) * 100.0, 1) if cap > 0 else 0.0

        budget_used = round(total_proc, 2)
        budget_pct = round((budget_used / self.total_budget) * 100.0, 1)

        return LPReorderResult(
            success=bool(res.success),
            status_message=str(res.message),
            orders=orders,
            shortages=shortages,
            total_procurement_cost=round(total_proc, 2),
            total_holding_cost=round(total_hold, 2),
            total_penalty_cost=round(total_pen, 2),
            total_projected_cost=round(total_proc + total_hold + total_pen, 2),
            budget_limit=self.total_budget,
            budget_used=budget_used,
            budget_utilization_pct=budget_pct,
            store_volume_utilization=vol_util,
        )
