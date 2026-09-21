from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

BASE = Path(__file__).resolve().parent

FORECAST_HORIZON = 30
BACKTEST_DAYS = 60

LEAD_TIME_DAYS = 7

# Service levels
SERVICE_LEVELS = {
    "90%": 1.28,
    "95%": 1.65,
    "99%": 2.33,
}

# Inventory cost assumptions
HOLDING_COST_PER_UNIT = 5.00       # $ / unit / year
STOCKOUT_COST_PER_UNIT = 5.00      # $ / unit
ORDERING_COST = 100.00             # $ / order


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    BASE / "sales.csv",
    parse_dates=["date"]
)

df = df.sort_values(["sku", "date"]).reset_index(drop=True)


# ============================================================
# 3. METRICS
# ============================================================

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.maximum(np.abs(y_true), 1e-8)

    return np.mean(
        np.abs((y_true - y_pred) / denominator)
    ) * 100


def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(y_true, y_pred)
    )


# ============================================================
# 4. SEASONAL MODEL
# ============================================================

def seasonal_fit(
    train,
    future_dates,
    use_promo=False,
    start_t=0
):
    """
    Seasonal demand model.

    Components:
    - Linear trend
    - Weekly seasonality
    - Monthly seasonality
    - Optional promotion effect

    This is the same forecasting idea used in the
    previous version.
    """

    y = train["units_sold"].astype(float).values

    t = np.arange(
        start_t,
        start_t + len(train)
    )

    # --------------------------------------------------------
    # Linear trend
    # --------------------------------------------------------

    slope, intercept = np.polyfit(
        t,
        y,
        1
    )

    # Trend fitted values
    trend_train = (
        intercept +
        slope * t
    )

    # Remove trend
    detrended = y - trend_train

    # --------------------------------------------------------
    # Weekly seasonality
    # --------------------------------------------------------

    train_dates = train["date"]

    dow = train_dates.dt.dayofweek

    weekly_factors = (
        pd.Series(
            detrended,
            index=train.index
        )
        .groupby(dow)
        .mean()
    )

    weekly_factors = weekly_factors.reindex(
        range(7),
        fill_value=0
    )

    # --------------------------------------------------------
    # Monthly seasonality
    # --------------------------------------------------------

    month = train_dates.dt.month

    monthly_factors = (
        pd.Series(
            detrended,
            index=train.index
        )
        .groupby(month)
        .mean()
    )

    monthly_factors = monthly_factors.reindex(
        range(1, 13),
        fill_value=0
    )

    # --------------------------------------------------------
    # Promotion effect
    # --------------------------------------------------------

    promo_effect = 0.0

    if use_promo and "promo" in train.columns:

        promo_mask = train["promo"].astype(int) == 1

        if promo_mask.sum() > 0:

            promo_effect = (
                train.loc[promo_mask, "units_sold"].mean()
                -
                train.loc[~promo_mask, "units_sold"].mean()
            )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    future_t = np.arange(
        start_t + len(train),
        start_t + len(train) + len(future_dates)
    )

    future_trend = (
        intercept +
        slope * future_t
    )

    future_dow = future_dates.dt.dayofweek

    future_month = future_dates.dt.month

    weekly_component = (
        future_dow.map(weekly_factors)
        .fillna(0)
        .values
    )

    monthly_component = (
        future_month.map(monthly_factors)
        .fillna(0)
        .values
    )

    prediction = (
        future_trend
        + weekly_component
        + monthly_component
    )

    # Promotion adjustment
    if use_promo:

        if "promo" in future_dates.to_frame().columns:
            pass

    return prediction, weekly_factors, monthly_factors, promo_effect


# ============================================================
# 5. SEASONAL + PROMO MODEL
# ============================================================

def seasonal_promo_forecast(
    train,
    future_dates,
    start_t=0
):
    """
    Seasonal model with promotion adjustment.
    """

    prediction, weekly_factors, monthly_factors, promo_effect = seasonal_fit(
        train=train,
        future_dates=future_dates,
        use_promo=True,
        start_t=start_t
    )

    # Determine promotion status
    if "promo" in future_dates.columns:
        future_promo = future_dates["promo"].astype(int).values
    else:
        future_promo = np.zeros(
            len(future_dates),
            dtype=int
        )

    prediction = (
        prediction
        + future_promo * promo_effect
    )

    return prediction


# ============================================================
# 6. BASELINE SEASONAL MODEL
# ============================================================

def seasonal_baseline_forecast(
    train,
    future_dates,
    start_t=0
):
    prediction, _, _, _ = seasonal_fit(
        train=train,
        future_dates=future_dates,
        use_promo=False,
        start_t=start_t
    )

    return prediction


# ============================================================
# 7. LINEAR REGRESSION MODEL
# ============================================================

def linear_regression_forecast(
    train,
    future_dates
):
    """
    Simple ML model.

    Features:
    - time index
    - day of week
    - month
    - promotion
    """

    train = train.copy()

    train["t"] = np.arange(len(train))

    train["dow"] = train["date"].dt.dayofweek

    train["month"] = train["date"].dt.month

    train["promo"] = train["promo"].astype(int)

    features = [
        "t",
        "dow",
        "month",
        "promo",
    ]

    model = LinearRegression()

    model.fit(
        train[features],
        train["units_sold"]
    )

    future = future_dates.copy()

    future["t"] = np.arange(
        len(train),
        len(train) + len(future)
    )

    future["dow"] = future["date"].dt.dayofweek

    future["month"] = future["date"].dt.month

    if "promo" not in future.columns:
        future["promo"] = 0

    future["promo"] = future["promo"].astype(int)

    prediction = model.predict(
        future[features]
    )

    return prediction


# ============================================================
# 8. BACKTEST MODELS
# ============================================================

model_results = []

sku_model_predictions = []

skus = sorted(df["sku"].unique())


for sku in skus:

    sku_df = (
        df[df["sku"] == sku]
        .sort_values("date")
        .reset_index(drop=True)
    )

    train = sku_df.iloc[:-BACKTEST_DAYS].copy()

    test = sku_df.iloc[-BACKTEST_DAYS:].copy()

    test_dates = test[
        ["date", "promo"]
    ].copy()

    start_t = 0

    # --------------------------------------------------------
    # Seasonal Baseline
    # --------------------------------------------------------

    pred_baseline = seasonal_baseline_forecast(
        train,
        test_dates,
        start_t=start_t
    )

    baseline_mape = mape(
        test["units_sold"],
        pred_baseline
    )

    baseline_rmse = rmse(
        test["units_sold"],
        pred_baseline
    )

    # --------------------------------------------------------
    # Seasonal + Promo
    # --------------------------------------------------------

    pred_promo = seasonal_promo_forecast(
        train,
        test_dates,
        start_t=start_t
    )

    promo_mape = mape(
        test["units_sold"],
        pred_promo
    )

    promo_rmse = rmse(
        test["units_sold"],
        pred_promo
    )

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    pred_lr = linear_regression_forecast(
        train,
        test_dates
    )

    lr_mape = mape(
        test["units_sold"],
        pred_lr
    )

    lr_rmse = rmse(
        test["units_sold"],
        pred_lr
    )

    # --------------------------------------------------------
    # Store model results
    # --------------------------------------------------------

    model_results.append({
        "sku": sku,
        "model": "Seasonal Baseline",
        "mape": baseline_mape,
        "rmse": baseline_rmse
    })

    model_results.append({
        "sku": sku,
        "model": "Seasonal + Promo",
        "mape": promo_mape,
        "rmse": promo_rmse
    })

    model_results.append({
        "sku": sku,
        "model": "Linear Regression",
        "mape": lr_mape,
        "rmse": lr_rmse
    })


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

model_results_df = pd.DataFrame(
    model_results
)

overall_model_comparison = (
    model_results_df
    .groupby("model")
    .agg({
        "mape": "mean",
        "rmse": "mean"
    })
    .reset_index()
)

overall_model_comparison = (
    overall_model_comparison
    .sort_values("mape")
    .reset_index(drop=True)
)

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for _, row in overall_model_comparison.iterrows():

    print(
        f"{row['model']:<25}"
        f" MAPE: {row['mape']:.2f}%"
        f"   RMSE: {row['rmse']:.2f}"
    )

best_overall_model = (
    overall_model_comparison
    .iloc[0]["model"]
)

print(
    f"\nBest model based on MAPE: "
    f"{best_overall_model}"
)


# ============================================================
# 10. BEST MODEL FOR EACH SKU
# ============================================================

best_model_by_sku = (
    model_results_df
    .sort_values(["sku", "mape"])
    .groupby("sku")
    .first()
    .reset_index()
)

print()
print("=" * 60)
print("BEST MODEL BY SKU")
print("=" * 60)

for _, row in best_model_by_sku.iterrows():

    print(
        f"{row['sku']}: "
        f"{row['model']} "
        f"(MAPE {row['mape']:.2f}%)"
    )


# ============================================================
# 11. 30-DAY FORECAST
# ============================================================

forecast_rows = []

inventory_rows = []


for _, sku_info in best_model_by_sku.iterrows():

    sku = sku_info["sku"]

    selected_model = sku_info["model"]

    sku_df = (
        df[df["sku"] == sku]
        .sort_values("date")
        .reset_index(drop=True)
    )

    train = sku_df.copy()

    last_date = train["date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    # --------------------------------------------------------
    # Future promotion assumption
    #
    # Since future promotion information is unknown,
    # we assume no promotion for the 30-day forecast.
    # --------------------------------------------------------

    future = pd.DataFrame({
        "date": future_dates,
        "promo": 0
    })

    # --------------------------------------------------------
    # Forecast using selected model
    # --------------------------------------------------------

    if selected_model == "Seasonal Baseline":

        predictions = seasonal_baseline_forecast(
            train,
            future,
            start_t=0
        )

    elif selected_model == "Seasonal + Promo":

        predictions = seasonal_promo_forecast(
            train,
            future,
            start_t=0
        )

    else:

        predictions = linear_regression_forecast(
            train,
            future
        )

    predictions = np.maximum(
        predictions,
        0
    )

    # --------------------------------------------------------
    # Store forecast
    # --------------------------------------------------------

    for date, prediction in zip(
        future_dates,
        predictions
    ):

        forecast_rows.append({
            "sku": sku,
            "date": date,
            "forecast_units": prediction,
            "model": selected_model
        })

    # --------------------------------------------------------
    # Inventory demand statistics
    # --------------------------------------------------------

    avg_daily_demand = float(
        np.mean(predictions)
    )

    # Use recent actual demand volatility
    recent_std = float(
        train["units_sold"]
        .tail(60)
        .std()
    )

    annual_demand = (
        avg_daily_demand * 365
    )

    # --------------------------------------------------------
    # Inventory calculations for each service level
    # --------------------------------------------------------

    for service_level, z in SERVICE_LEVELS.items():

        # ----------------------------------------------------
        # Safety Stock
        # ----------------------------------------------------

        safety_stock = (
            z
            * recent_std
            * math.sqrt(LEAD_TIME_DAYS)
        )

        # ----------------------------------------------------
        # Lead-time demand
        # ----------------------------------------------------

        lead_time_demand = (
            avg_daily_demand
            * LEAD_TIME_DAYS
        )

        # ----------------------------------------------------
        # Reorder Point
        # ----------------------------------------------------

        reorder_point = (
            lead_time_demand
            + safety_stock
        )

        # ====================================================
        # EOQ
        # ====================================================

        eoq = math.sqrt(
            (
                2
                * annual_demand
                * ORDERING_COST
            )
            / HOLDING_COST_PER_UNIT
        )

        # ----------------------------------------------------
        # Number of orders per year
        # ----------------------------------------------------

        orders_per_year = (
            annual_demand / eoq
        )

        # ----------------------------------------------------
        # Average cycle inventory
        # ----------------------------------------------------

        cycle_inventory = eoq / 2

        # ----------------------------------------------------
        # Average inventory
        #
        # EOQ/2 = cycle stock
        # Safety stock = uncertainty buffer
        # ----------------------------------------------------

        average_inventory = (
            cycle_inventory
            + safety_stock
        )

        # ----------------------------------------------------
        # Holding Cost
        # ----------------------------------------------------

        holding_cost = (
            average_inventory
            * HOLDING_COST_PER_UNIT
        )

        # ----------------------------------------------------
        # Ordering Cost
        # ----------------------------------------------------

        ordering_cost = (
            orders_per_year
            * ORDERING_COST
        )

        # ----------------------------------------------------
        # Expected shortage
        #
        # Demand during lead time is approximated
        # by a normal distribution.
        # ----------------------------------------------------

        lead_time_std = (
            recent_std
            * math.sqrt(LEAD_TIME_DAYS)
        )

        if lead_time_std > 0:

            # Standard normal PDF
            phi = (
                1
                / math.sqrt(2 * math.pi)
                * math.exp(-0.5 * z * z)
            )

            # Standard normal tail probability
            tail_probability = (
                0.5
                * math.erfc(
                    z / math.sqrt(2)
                )
            )

            expected_shortage_per_cycle = (
                lead_time_std
                * (
                    phi
                    - z * tail_probability
                )
            )

        else:

            expected_shortage_per_cycle = 0.0

        # ----------------------------------------------------
        # Annual expected shortage
        # ----------------------------------------------------

        expected_annual_shortage = (
            expected_shortage_per_cycle
            * orders_per_year
        )

        # ----------------------------------------------------
        # Stockout Cost
        # ----------------------------------------------------

        stockout_cost = (
            expected_annual_shortage
            * STOCKOUT_COST_PER_UNIT
        )

        # ----------------------------------------------------
        # Total Inventory Cost
        # ----------------------------------------------------

        total_inventory_cost = (
            holding_cost
            + ordering_cost
            + stockout_cost
        )

        # ----------------------------------------------------
        # Order-up-to level
        #
        # We keep the previous concept:
        # lead-time demand + review-period demand
        # + safety stock.
        #
        # This is retained so we don't change the
        # previous inventory logic unnecessarily.
        # ----------------------------------------------------

        review_period_demand = (
            avg_daily_demand * 7
        )

        order_up_to_level = (
            lead_time_demand
            + review_period_demand
            + safety_stock
        )

        inventory_rows.append({

            "sku": sku,

            "category": train["category"].iloc[0],

            "model": selected_model,

            "service_level": service_level,

            "avg_daily_demand": avg_daily_demand,

            "demand_std": recent_std,

            "annual_demand": annual_demand,

            "lead_time_demand": lead_time_demand,

            "safety_stock": safety_stock,

            "reorder_point": reorder_point,

            "eoq": eoq,

            "orders_per_year": orders_per_year,

            "cycle_inventory": cycle_inventory,

            "average_inventory": average_inventory,

            "order_up_to_level": order_up_to_level,

            "holding_cost": holding_cost,

            "ordering_cost": ordering_cost,

            "expected_annual_shortage": expected_annual_shortage,

            "stockout_cost": stockout_cost,

            "total_inventory_cost": total_inventory_cost
        })


# ============================================================
# 12. SAVE FORECAST
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)

forecast_df.to_csv(
    BASE / "forecast_30d.csv",
    index=False
)


# ============================================================
# 13. INVENTORY OPTIMIZATION DATA
# ============================================================

inventory_df = pd.DataFrame(
    inventory_rows
)

inventory_df.to_csv(
    BASE / "inventory_optimization.csv",
    index=False
)


# ============================================================
# 14. SERVICE LEVEL ANALYSIS
# ============================================================

service_level_analysis = (
    inventory_df
    .groupby("service_level")
    .agg({
        "safety_stock": "sum",
        "average_inventory": "sum",
        "holding_cost": "sum",
        "ordering_cost": "sum",
        "expected_annual_shortage": "sum",
        "stockout_cost": "sum",
        "total_inventory_cost": "sum"
    })
    .reset_index()
)

service_level_analysis = (
    service_level_analysis
    .sort_values("service_level")
)

service_level_analysis.to_csv(
    BASE / "service_level_analysis.csv",
    index=False
)


# ============================================================
# 15. PRINT SERVICE LEVEL ANALYSIS
# ============================================================

print()
print("=" * 60)
print("SERVICE LEVEL ANALYSIS")
print("=" * 60)

print(
    service_level_analysis.to_string(
        index=False
    )
)


# ============================================================
# 16. PRINT 95% INVENTORY PLAN
# ============================================================

inventory_95 = (
    inventory_df[
        inventory_df["service_level"] == "95%"
    ]
    .copy()
)

print()
print("=" * 60)
print("95% SERVICE LEVEL INVENTORY PLAN")
print("=" * 60)

display_columns = [
    "sku",
    "category",
    "model",
    "avg_daily_demand",
    "demand_std",
    "annual_demand",
    "lead_time_demand",
    "safety_stock",
    "reorder_point",
    "eoq",
    "orders_per_year",
    "average_inventory",
    "order_up_to_level",
    "holding_cost",
    "ordering_cost",
    "stockout_cost",
    "total_inventory_cost"
]

print(
    inventory_95[
        display_columns
    ].round(2).to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE MODEL COMPARISON
# ============================================================

overall_model_comparison.to_csv(
    BASE / "model_comparison.csv",
    index=False
)

model_results_df.to_csv(
    BASE / "model_comparison_by_sku.csv",
    index=False
)


# ============================================================
# 18. MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    overall_model_comparison["model"],
    overall_model_comparison["mape"]
)

plt.ylabel("MAPE (%)")

plt.title(
    "Model Comparison by Average MAPE"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    BASE / "model_comparison.png"
)

plt.close()


# ============================================================
# 19. SERVICE LEVEL VS COST CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    service_level_analysis["service_level"],
    service_level_analysis["total_inventory_cost"],
    marker="o"
)

plt.xlabel("Service Level")

plt.ylabel("Total Inventory Cost")

plt.title(
    "Service Level vs Total Inventory Cost"
)

plt.tight_layout()

plt.savefig(
    BASE / "service_level_vs_cost.png"
)

plt.close()


# ============================================================
# 20. EOQ CHART
# ============================================================

eoq_95 = (
    inventory_df[
        inventory_df["service_level"] == "95%"
    ]
)

plt.figure(figsize=(8, 5))

plt.bar(
    eoq_95["sku"],
    eoq_95["eoq"]
)

plt.xlabel("SKU")

plt.ylabel("EOQ (units)")

plt.title(
    "Economic Order Quantity by SKU"
)

plt.tight_layout()

plt.savefig(
    BASE / "eoq_by_sku.png"
)

plt.close()


# ============================================================
# 21. INVENTORY COST COMPONENT CHART
# ============================================================

cost_plot = inventory_95.set_index("sku")[
    [
        "holding_cost",
        "ordering_cost",
        "stockout_cost"
    ]
]

cost_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.ylabel("Annual Cost ($)")

plt.title(
    "Inventory Cost Components at 95% Service Level"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    BASE / "inventory_cost_components.png"
)

plt.close()


# ============================================================
# 22. RESULTS JSON
# ============================================================

results = {

    "skus": int(df["sku"].nunique()),

    "days": int(
        df["date"].nunique()
    ),

    "backtest_days": BACKTEST_DAYS,

    "best_overall_model": best_overall_model,

    "model_comparison": (
        overall_model_comparison
        .round(2)
        .to_dict(orient="records")
    ),

    "forecast_horizon_days": FORECAST_HORIZON,

    "lead_time_days": LEAD_TIME_DAYS,

    "service_levels": list(
        SERVICE_LEVELS.keys()
    ),

    "inventory_cost_assumptions": {

        "holding_cost_per_unit_per_year":
            HOLDING_COST_PER_UNIT,

        "stockout_cost_per_unit":
            STOCKOUT_COST_PER_UNIT,

        "ordering_cost_per_order":
            ORDERING_COST
    },

    "eoq_added": True
}


with open(
    BASE / "results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"SKUs: {df['sku'].nunique()}"
)

print(
    f"Forecast horizon: "
    f"{FORECAST_HORIZON} days"
)

print(
    f"Lead time: "
    f"{LEAD_TIME_DAYS} days"
)

print(
    f"Best overall model: "
    f"{best_overall_model}"
)

print(
    "\n95% service level:"
)

print(
    f"Total EOQ: "
    f"{inventory_95['eoq'].sum():.2f}"
)

print(
    f"Total annual demand: "
    f"{inventory_95['annual_demand'].sum():.2f}"
)

print(
    f"Total annual ordering cost: "
    f"${inventory_95['ordering_cost'].sum():,.2f}"
)

print(
    f"Total annual holding cost: "
    f"${inventory_95['holding_cost'].sum():,.2f}"
)

print(
    f"Total annual stockout cost: "
    f"${inventory_95['stockout_cost'].sum():,.2f}"
)

print(
    f"Total annual inventory cost: "
    f"${inventory_95['total_inventory_cost'].sum():,.2f}"
)

print()
print("Files generated:")
print(" - forecast_30d.csv")
print(" - inventory_optimization.csv")
print(" - service_level_analysis.csv")
print(" - model_comparison.csv")
print(" - model_comparison_by_sku.csv")
print(" - results.json")
print(" - model_comparison.png")
print(" - service_level_vs_cost.png")
print(" - eoq_by_sku.png")
print(" - inventory_cost_components.png")