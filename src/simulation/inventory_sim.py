"""
src/simulation/inventory_sim.py

Discrete-event inventory simulator comparing policies over real test horizons:
Policy A: Quantile Forecast + LP Optimizer (Our Solution)
Policy B: 7-Day Moving Average Baseline Policy (Industry Baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from src.data.generator import ITEM_CATALOG
from src.optimization.lp_solver import InventoryLPSolver, ItemOrderContext
from src.optimization.safety_stock import compute_reorder_point


@dataclass
class PolicySimulationResult:
    policy_name: str
    total_demand_units: float
    total_fulfilled_units: float
    total_lost_sales_units: float
    stockout_rate_pct: float
    service_level_pct: float
    total_holding_cost: float
    total_stockout_penalty: float
    total_procurement_cost: float
    total_operating_cost: float   # Holding + Stockout penalty


class InventorySimulator:
    """Simulates daily stock levels, sales, stockouts, and replenishment cycles."""

    def __init__(
        self,
        test_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        solver: InventoryLPSolver,
        initial_stock_units: float = 30.0,
    ):
        self.test_df = test_df.copy()
        self.forecast_df = forecast_df.copy()
        self.solver = solver
        self.initial_stock = initial_stock_units
        self.pairs: List[Tuple[str, str]] = sorted(
            list(set(zip(test_df["store_id"], test_df["item_id"])))
        )

    def run_simulation(self) -> Dict[str, PolicySimulationResult]:
        logger.info(f"Running 30-day inventory simulation over {len(self.pairs)} store-item series ...")
        res_opt = self._simulate_policy(use_lp_optimizer=True)
        res_base = self._simulate_policy(use_lp_optimizer=False)

        savings = res_base.total_operating_cost - res_opt.total_operating_cost
        pct_savings = (savings / res_base.total_operating_cost) * 100 if res_base.total_operating_cost > 0 else 0

        logger.info(
            f"Simulation Results:\n"
            f"  [Baseline Moving Avg] Stockout Rate: {res_base.stockout_rate_pct:.1f}% | Op Cost: ${res_base.total_operating_cost:,.2f}\n"
            f"  [LP Quantile Optimal] Stockout Rate: {res_opt.stockout_rate_pct:.1f}% | Op Cost: ${res_opt.total_operating_cost:,.2f}\n"
            f"  ==> Net Operational Savings: ${savings:,.2f} ({pct_savings:.1f}% cost reduction)"
        )

        return {
            "LP_Quantile_Optimal": res_opt,
            "Baseline_Moving_Average": res_base,
        }

    def _simulate_policy(self, use_lp_optimizer: bool) -> PolicySimulationResult:
        policy_name = "LP_Quantile_Optimal" if use_lp_optimizer else "Baseline_Moving_Average"
        inventory: Dict[Tuple[str, str], float] = {p: self.initial_stock for p in self.pairs}
        pipeline_orders: Dict[Tuple[str, str], List[Tuple[int, float]]] = {p: [] for p in self.pairs}

        total_demand = 0.0
        total_fulfilled = 0.0
        total_lost = 0.0
        total_holding_cost = 0.0
        total_penalty_cost = 0.0
        total_procurement_cost = 0.0

        dates = sorted(self.test_df["date"].unique())

        for day_idx, current_date in enumerate(dates):
            # 1. Receive pipeline orders arriving today
            for pair in self.pairs:
                arrived = [qty for arrival_day, qty in pipeline_orders[pair] if arrival_day == day_idx]
                inventory[pair] += sum(arrived)
                pipeline_orders[pair] = [(ad, q) for ad, q in pipeline_orders[pair] if ad > day_idx]

            # 2. Reorder review every 7 days
            if day_idx % 7 == 0:
                if use_lp_optimizer:
                    # Prepare LP context with dynamic safety stocks from Quantile P90-P50
                    lp_items: List[ItemOrderContext] = []
                    for s_id, i_id in self.pairs:
                        meta = ITEM_CATALOG[i_id]
                        # Lookup predicted quantiles for this date or mean over week
                        f_slice = self.forecast_df[
                            (self.forecast_df["store_id"] == s_id) & (self.forecast_df["item_id"] == i_id)
                        ]
                        p50 = float(f_slice["pred_q50"].mean()) if not f_slice.empty else 20.0
                        p90 = float(f_slice["pred_q90"].mean()) if not f_slice.empty else 30.0

                        rop_info = compute_reorder_point(
                            pred_p50_daily=p50,
                            pred_p90_daily=p90,
                            lead_time_days=meta.lead_time_days,
                        )

                        lp_items.append(
                            ItemOrderContext(
                                store_id=s_id,
                                item_id=i_id,
                                current_inventory=inventory[(s_id, i_id)],
                                target_rop=rop_info["reorder_point"],
                                unit_cost=meta.unit_cost,
                                holding_cost_daily=meta.holding_cost_daily,
                                stockout_penalty=meta.stockout_penalty,
                                unit_volume_m3=meta.unit_volume_m3,
                            )
                        )

                    lp_res = self.solver.solve(lp_items)
                    for (s_id, i_id), q_val in lp_res.orders.items():
                        if q_val > 0:
                            lead_time = ITEM_CATALOG[i_id].lead_time_days
                            pipeline_orders[(s_id, i_id)].append((day_idx + lead_time, q_val))
                            total_procurement_cost += q_val * ITEM_CATALOG[i_id].unit_cost
                else:
                    # Baseline: simple 7-day historical moving average reorder without LP optimization
                    for s_id, i_id in self.pairs:
                        meta = ITEM_CATALOG[i_id]
                        hist_slice = self.test_df[
                            (self.test_df["store_id"] == s_id) & (self.test_df["item_id"] == i_id) & (self.test_df["date"] <= current_date)
                        ]
                        base_demand = float(hist_slice["sales"].tail(7).mean()) if not hist_slice.empty else 20.0
                        target = base_demand * 7.0  # naive 7-day cover
                        order_qty = max(0.0, target - inventory[(s_id, i_id)])
                        if order_qty > 0:
                            lead_time = meta.lead_time_days
                            pipeline_orders[(s_id, i_id)].append((day_idx + lead_time, order_qty))
                            total_procurement_cost += order_qty * meta.unit_cost

            # 3. Realize customer demand on current day
            day_records = self.test_df[self.test_df["date"] == current_date]
            for _, row in day_records.iterrows():
                pair = (row["store_id"], row["item_id"])
                meta = ITEM_CATALOG[row["item_id"]]
                dem = float(row["sales"])
                total_demand += dem

                on_hand = inventory[pair]
                if on_hand >= dem:
                    inventory[pair] -= dem
                    total_fulfilled += dem
                else:
                    # Stockout
                    fulfilled = on_hand
                    lost = dem - on_hand
                    inventory[pair] = 0.0
                    total_fulfilled += fulfilled
                    total_lost += lost
                    total_penalty_cost += lost * meta.stockout_penalty

                # End of day holding cost
                total_holding_cost += inventory[pair] * meta.holding_cost_daily

        stockout_rate = (total_lost / total_demand * 100.0) if total_demand > 0 else 0.0
        service_level = 100.0 - stockout_rate

        return PolicySimulationResult(
            policy_name=policy_name,
            total_demand_units=round(total_demand, 1),
            total_fulfilled_units=round(total_fulfilled, 1),
            total_lost_sales_units=round(total_lost, 1),
            stockout_rate_pct=round(stockout_rate, 2),
            service_level_pct=round(service_level, 2),
            total_holding_cost=round(total_holding_cost, 2),
            total_stockout_penalty=round(total_penalty_cost, 2),
            total_procurement_cost=round(total_procurement_cost, 2),
            total_operating_cost=round(total_holding_cost + total_penalty_cost, 2),
        )
