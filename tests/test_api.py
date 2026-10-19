"""
tests/test_api.py

Async tests for FastAPI forecasting and inventory optimization endpoints.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.fixture(scope="module")
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_forecast_quantiles_endpoint(client: AsyncClient):
    payload = {
        "store_id": "STORE_A",
        "item_id": "ITEM_01",
        "horizon_days": 7,
    }
    resp = await client.post("/forecast/quantiles", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["store_id"] == "STORE_A"
    assert len(data["daily_predictions"]) == 7
    assert data["dynamic_safety_stock"] > 0
    assert data["reorder_point"] > 0


@pytest.mark.asyncio
async def test_inventory_optimize_endpoint(client: AsyncClient):
    payload = {
        "inventory_levels": [
            {"store_id": "STORE_A", "item_id": "ITEM_01", "current_inventory": 10.0},
            {"store_id": "STORE_A", "item_id": "ITEM_02", "current_inventory": 2.0},
        ],
        "total_budget": 5000.0,
    }
    resp = await client.post("/inventory/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_procurement_cost"] <= 5000.0
    assert len(data["recommendations"]) == 2


@pytest.mark.asyncio
async def test_inventory_simulate_endpoint(client: AsyncClient):
    resp = await client.post("/inventory/simulate")
    assert resp.status_code == 200
    data = resp.json()
    assert "optimal_stockout_rate_pct" in data
    assert "baseline_stockout_rate_pct" in data
    assert "net_dollar_savings" in data
