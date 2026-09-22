from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parent


# ============================================================
# TOOL 1: GET INVENTORY DATA FOR A SKU
# ============================================================

def get_inventory_data(sku: str):
    """
    Get inventory optimization results for a specific SKU.
    """

    inventory_path = BASE / "inventory_recommendations.csv"

    inventory_df = pd.read_csv(inventory_path)

    sku_data = inventory_df[
        inventory_df["sku"].str.upper() == sku.upper()
    ].copy()

    if sku_data.empty:
        return {
            "success": False,
            "message": f"No inventory data found for {sku}."
        }

    # Use 95% service level if available
    if "95%" in sku_data["service_level"].astype(str).values:
        row = sku_data[
            sku_data["service_level"].astype(str) == "95%"
        ].iloc[0]
    else:
        row = sku_data.iloc[0]

    return {
        "success": True,
        "sku": str(row["sku"]),
        "service_level": str(row["service_level"]),
        "avg_daily_demand": float(row["avg_daily_demand"]),
        "demand_std": float(row["demand_std"]),
        "annual_demand": float(row["annual_demand"]),
        "lead_time_demand": float(row["lead_time_demand"]),
        "safety_stock": float(row["safety_stock"]),
        "reorder_point": float(row["reorder_point"]),
        "order_up_to_level": float(row["order_up_to_level"]),
        "average_inventory": float(row["average_inventory"]),
        "eoq": float(row["eoq"]),
        "eoq_order_interval_days": float(
            row["eoq_order_interval_days"]
        ),
        "holding_cost": float(row["holding_cost"]),
        "ordering_cost": float(row["ordering_cost"]),
        "stockout_cost": float(row["stockout_cost"]),
        "total_inventory_cost": float(
            row["total_inventory_cost"]
        ),
        "forecast_model": str(row["model"])
    }


# ============================================================
# TOOL 2: GET MODEL PERFORMANCE FOR A SKU
# ============================================================

def get_model_performance(sku: str):
    """
    Get forecasting model performance for a specific SKU.
    """

    model_path = BASE / "model_comparison.csv"

    model_df = pd.read_csv(model_path)

    sku_data = model_df[
        model_df["sku"].str.upper() == sku.upper()
    ].copy()

    if sku_data.empty:
        return {
            "success": False,
            "message": f"No model comparison data found for {sku}."
        }

    row = sku_data.iloc[0]

    models = {
        "Seasonal Baseline": float(row["seasonal_mape"]),
        "Seasonal + Promo": float(row["promo_mape"]),
        "Linear Regression": float(row["regression_mape"])
    }

    best_model = min(
        models,
        key=models.get
    )

    return {
        "success": True,
        "sku": str(row["sku"]),
        "model_mape": models,
        "best_model": best_model,
        "best_mape": models[best_model]
    }