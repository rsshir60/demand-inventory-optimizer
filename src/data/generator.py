"""
src/data/generator.py

Simulates multi-store, multi-item hierarchical daily demand with realistic retail dynamics:
- Weekly seasonality (weekend surges)
- Monthly & annual trend
- Promotional events (price elasticity spikes)
- Item metadata: Unit cost, retail price, holding cost, stockout penalty, volume, supplier lead time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ItemMetadata:
    item_id: str
    category: str
    unit_cost: float       # Procurement cost per unit ($)
    unit_price: float      # Retail selling price ($)
    holding_cost_daily: float  # Storage / capital cost per day ($)
    stockout_penalty: float    # Lost profit + customer goodwill penalty ($)
    unit_volume_m3: float      # Warehouse footprint (cubic meters)
    lead_time_days: int        # Supplier replenishment lead time


ITEM_CATALOG: dict[str, ItemMetadata] = {
    "ITEM_01": ItemMetadata("ITEM_01", "Beverages", unit_cost=3.0, unit_price=6.0, holding_cost_daily=0.04, stockout_penalty=8.0, unit_volume_m3=0.015, lead_time_days=2),
    "ITEM_02": ItemMetadata("ITEM_02", "Electronics", unit_cost=85.0, unit_price=160.0, holding_cost_daily=0.45, stockout_penalty=120.0, unit_volume_m3=0.040, lead_time_days=4),
    "ITEM_03": ItemMetadata("ITEM_03", "Apparel", unit_cost=15.0, unit_price=40.0, holding_cost_daily=0.10, stockout_penalty=35.0, unit_volume_m3=0.025, lead_time_days=3),
    "ITEM_04": ItemMetadata("ITEM_04", "Packaged_Food", unit_cost=4.5, unit_price=8.5, holding_cost_daily=0.05, stockout_penalty=10.0, unit_volume_m3=0.020, lead_time_days=2),
    "ITEM_05": ItemMetadata("ITEM_05", "Home_Goods", unit_cost=28.0, unit_price=65.0, holding_cost_daily=0.22, stockout_penalty=50.0, unit_volume_m3=0.060, lead_time_days=5),
}

STORES = ["STORE_A", "STORE_B", "STORE_C"]
STORE_MULTIPLIERS = {"STORE_A": 1.4, "STORE_B": 1.0, "STORE_C": 0.7}


def generate_hierarchical_demand(
    n_days: int = 730,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generates 2 years of daily sales for 3 stores x 5 items (15 series, 10,950 total rows).
    """
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(end="2026-10-20", periods=n_days, freq="D")
    records = []

    for store_id in STORES:
        s_mult = STORE_MULTIPLIERS[store_id]

        for item_id, meta in ITEM_CATALOG.items():
            # Base daily demand per item
            base_demand = {
                "ITEM_01": 55.0,
                "ITEM_02": 12.0,
                "ITEM_03": 25.0,
                "ITEM_04": 65.0,
                "ITEM_05": 16.0,
            }[item_id]

            for i, d in enumerate(dates):
                # 1. Day of week seasonality: Weekends (Sat/Sun) +35% to +50%
                dow = d.dayofweek
                dow_factor = 1.45 if dow in (5, 6) else (0.85 if dow == 0 else 1.0)

                # 2. Gentle annual trend + annual seasonality
                day_of_year = d.dayofyear
                annual_seasonality = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)
                trend = 1.0 + 0.08 * (i / n_days)

                # 3. Random promotions: ~5% of days have promotional discount
                is_promo = int(rng.random() < 0.05)
                promo_factor = 2.2 if is_promo else 1.0

                # Expected mean
                mu = base_demand * s_mult * dow_factor * annual_seasonality * trend * promo_factor
                # Overdispersed negative binomial / Poisson noise
                sales = max(0, int(rng.poisson(mu) + rng.normal(0, np.sqrt(mu) * 0.4)))

                records.append({
                    "date": d,
                    "store_id": store_id,
                    "item_id": item_id,
                    "sales": sales,
                    "is_promo": is_promo,
                    "unit_cost": meta.unit_cost,
                    "unit_price": meta.unit_price,
                    "holding_cost_daily": meta.holding_cost_daily,
                    "stockout_penalty": meta.stockout_penalty,
                    "unit_volume_m3": meta.unit_volume_m3,
                    "lead_time_days": meta.lead_time_days,
                })

    df = pd.DataFrame(records).sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    logger.info(
        f"Generated hierarchical demand: {len(df):,} rows | "
        f"{df['store_id'].nunique()} stores | {df['item_id'].nunique()} items | "
        f"Mean daily sales: {df['sales'].mean():.1f}"
    )
    return df
