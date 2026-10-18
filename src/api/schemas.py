"""
src/api/schemas.py

Pydantic v2 schemas for the Demand Forecasting & Inventory Optimization service.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    store_id: str = Field(default="STORE_A", description="Store identifier")
    item_id: str = Field(default="ITEM_01", description="Item identifier")
    horizon_days: int = Field(default=7, ge=1, le=28, description="Forecast horizon in days")


class DailyQuantilePrediction(BaseModel):
    day_offset: int
    p10: float
    p50: float
    p90: float


class ForecastResponse(BaseModel):
    store_id: str
    item_id: str
    horizon_days: int
    daily_predictions: List[DailyQuantilePrediction]
    lead_time_days: int
    dynamic_safety_stock: float
    reorder_point: float


class StoreInventoryInput(BaseModel):
    store_id: str
    item_id: str
    current_inventory: float = Field(..., ge=0)


class ReorderOptimizeRequest(BaseModel):
    inventory_levels: List[StoreInventoryInput]
    total_budget: float = Field(default=35_000.0, gt=0)
    store_capacities: Optional[Dict[str, float]] = None


class RecommendedOrderItem(BaseModel):
    store_id: str
    item_id: str
    current_inventory: float
    target_rop: float
    recommended_order_units: float
    unit_cost: float
    subtotal_cost: float


class ReorderOptimizeResponse(BaseModel):
    success: bool
    status_message: str
    total_procurement_cost: float
    budget_limit: float
    budget_utilization_pct: float
    store_volume_utilization: Dict[str, float]
    recommendations: List[RecommendedOrderItem]


class SimulationComparisonResponse(BaseModel):
    baseline_stockout_rate_pct: float
    optimal_stockout_rate_pct: float
    stockout_reduction_pct: float
    baseline_operating_cost: float
    optimal_operating_cost: float
    net_dollar_savings: float
    operating_cost_reduction_pct: float
