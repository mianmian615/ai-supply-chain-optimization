
"""
forecast.py
AI-Driven Supply Chain Demand Forecasting & Inventory Optimization

Day 2:
1. Seasonal forecasting model
2. Seasonal + Promotion model
3. Linear Regression model
4. Backtest and compare MAPE / RMSE
5. Select the best model
6. Forecast next 30 days

Day 3:
1. Demand forecasting
2. Model comparison
3. Promotion feature
4. Safety stock
5. Reorder point
6. Order-up-to level
7. Service-level analysis
8. Inventory cost optimization
9. EOQ analysis
"""

import pathlib
import json
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# ============================================================
# 1. BASIC SETTINGS
# ============================================================

BASE = pathlib.Path(__file__).resolve().parent

HORIZON = 30
HOLDOUT = 60
LEAD = 7

# Default service level
Z = 1.65   # 95% service level


# ============================================================
# 2. SERVICE LEVEL SCENARIOS
# ============================================================

SERVICE_LEVELS = {
    "90%": 1.28,
    "95%": 1.65,
    "99%": 2.33
}


# ============================================================
# 3. INVENTORY COST ASSUMPTIONS
# ============================================================

# Annual holding cost per unit
HOLDING_COST_PER_UNIT_PER_YEAR = 5.00

# Cost of one unit of expected shortage
STOCKOUT_COST_PER_UNIT = 5.00

# Fixed cost per order
ORDERING_COST_PER_ORDER = 100.00


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(
    BASE / "sales.csv",
    parse_dates=["date"]
)

df = (
    df
    .sort_values(["sku", "date"])
    .reset_index(drop=True)
)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

def create_features(data):
    """
    Create features for Linear Regression.

    Features:
    - t: sequential time index
    - dow: day of week
    - month: month
    - promo: promotion indicator
    """

    data = data.copy()

    # Sequential time index
    data["t"] = np.arange(len(data))

    # Day of week
    # Monday = 0 ... Sunday = 6
    data["dow"] = data["date"].dt.dayofweek

    # Month
    data["month"] = data["date"].dt.month

    return data


# ============================================================
# 6. SEASONAL + PROMOTION MODEL
# ============================================================

def seasonal_fit(train):
    """
    Fit:
        trend
        weekly seasonality
        monthly seasonality
        promotion effect
    """

    # Time index
    t = np.arange(len(train))

    # Actual demand
    y = train["units_sold"].values.astype(float)

    # --------------------------------------------------------
    # Linear trend
    # --------------------------------------------------------

    b, a = np.polyfit(t, y, 1)

    # Remove trend
    detr = y - (a + b * t)

    # --------------------------------------------------------
    # Weekly seasonality
    # --------------------------------------------------------

    wk = (
        pd.Series(detr)
        .groupby(
            train["date"].dt.dayofweek.values
        )
        .mean()
    )

    # Remove weekly effect
    weekly_values = (
        wk
        .reindex(
            train["date"].dt.dayofweek.values
        )
        .values
    )

    detr2 = detr - weekly_values

    # --------------------------------------------------------
    # Monthly seasonality
    # --------------------------------------------------------

    mo = (
        pd.Series(detr2)
        .groupby(
            train["date"].dt.month.values
        )
        .mean()
    )

    # --------------------------------------------------------
    # Promotion effect
    # --------------------------------------------------------

    promo_mean = train.loc[
        train["promo"] == 1,
        "units_sold"
    ].mean()

    non_promo_mean = train.loc[
        train["promo"] == 0,
        "units_sold"
    ].mean()

    promo_effect = promo_mean - non_promo_mean

    return (
        a,
        b,
        wk,
        mo,
        promo_effect
    )


def seasonal_predict(
    a,
    b,
    wk,
    mo,
    promo_effect,
    dates,
    promo,
    t0
):
    """
    Generate predictions using:

    trend
    + weekly seasonality
    + monthly seasonality
    + promotion effect
    """

    dates = pd.Series(
        pd.to_datetime(dates)
    ).reset_index(drop=True)

    # Time index for prediction period
    t = np.arange(
        t0,
        t0 + len(dates)
    )

    # Trend
    base = a + b * t

    # Calendar features
    dow = dates.dt.dayofweek.values
    month = dates.dt.month.values

    # Seasonal effects
    weekly_effect = (
        wk
        .reindex(dow)
        .values
    )

    monthly_effect = (
        mo
        .reindex(month)
        .values
    )

    # Promotion flag
    promo = np.asarray(promo)

    # Final prediction
    prediction = (
        base
        + weekly_effect
        + monthly_effect
        + promo * promo_effect
    )

    # Demand cannot be negative
    return np.clip(
        prediction,
        0,
        None
    )


# ============================================================
# 7. LINEAR REGRESSION MODEL
# ============================================================

def regression_fit(train):
    """
    Train Linear Regression.

    Features:
        t
        day of week
        month
        promotion
    """

    train = create_features(train)

    X = train[
        [
            "t",
            "dow",
            "month",
            "promo"
        ]
    ]

    y = train["units_sold"]

    model = LinearRegression()

    model.fit(X, y)

    return model


def regression_predict(
    model,
    dates,
    promo,
    t0
):
    """
    Generate predictions using Linear Regression.
    """

    dates = pd.Series(
        pd.to_datetime(dates)
    ).reset_index(drop=True)

    future = pd.DataFrame()

    future["t"] = np.arange(
        t0,
        t0 + len(dates)
    )

    future["dow"] = (
        dates.dt.dayofweek.values
    )

    future["month"] = (
        dates.dt.month.values
    )

    future["promo"] = np.asarray(
        promo
    )

    X = future[
        [
            "t",
            "dow",
            "month",
            "promo"
        ]
    ]

    prediction = model.predict(X)

    return np.clip(
        prediction,
        0,
        None
    )


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def calculate_mape(actual, predicted):
    """
    MAPE:
    Mean Absolute Percentage Error
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.mean(
            np.abs(
                (actual - predicted)
                / np.clip(actual, 1, None)
            )
        )
        * 100
    )


def calculate_rmse(actual, predicted):
    """
    RMSE:
    Root Mean Squared Error
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return float(
        np.sqrt(
            np.mean(
                (actual - predicted) ** 2
            )
        )
    )


# ============================================================
# 9. BACKTEST ALL MODELS
# ============================================================

seasonal_mape = {}
promo_mape = {}
regression_mape = {}

seasonal_rmse = {}
promo_rmse = {}
regression_rmse = {}


for sku, g in df.groupby("sku"):

    g = (
        g
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    train = g.iloc[:-HOLDOUT]

    test = g.iloc[-HOLDOUT:]

    # ========================================================
    # MODEL 1
    # Seasonal Baseline
    # ========================================================

    (
        a,
        b,
        wk,
        mo,
        promo_effect
    ) = seasonal_fit(train)

    # Ignore promotions
    no_promo = np.zeros(
        len(test)
    )

    pred_seasonal = seasonal_predict(
        a,
        b,
        wk,
        mo,
        promo_effect,
        test["date"],
        no_promo,
        len(train)
    )

    seasonal_mape[sku] = calculate_mape(
        test["units_sold"],
        pred_seasonal
    )

    seasonal_rmse[sku] = calculate_rmse(
        test["units_sold"],
        pred_seasonal
    )

    # ========================================================
    # MODEL 2
    # Seasonal + Promotion
    # ========================================================

    pred_promo = seasonal_predict(
        a,
        b,
        wk,
        mo,
        promo_effect,
        test["date"],
        test["promo"].values,
        len(train)
    )

    promo_mape[sku] = calculate_mape(
        test["units_sold"],
        pred_promo
    )

    promo_rmse[sku] = calculate_rmse(
        test["units_sold"],
        pred_promo
    )

    # ========================================================
    # MODEL 3
    # Linear Regression
    # ========================================================

    regression_model = regression_fit(
        train
    )

    pred_regression = regression_predict(
        regression_model,
        test["date"],
        test["promo"].values,
        len(train)
    )

    regression_mape[sku] = calculate_mape(
        test["units_sold"],
        pred_regression
    )

    regression_rmse[sku] = calculate_rmse(
        test["units_sold"],
        pred_regression
    )


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

avg_seasonal_mape = np.mean(
    list(seasonal_mape.values())
)

avg_promo_mape = np.mean(
    list(promo_mape.values())
)

avg_regression_mape = np.mean(
    list(regression_mape.values())
)


avg_seasonal_rmse = np.mean(
    list(seasonal_rmse.values())
)

avg_promo_rmse = np.mean(
    list(promo_rmse.values())
)

avg_regression_rmse = np.mean(
    list(regression_rmse.values())
)


model_scores = {
    "Seasonal Baseline": avg_seasonal_mape,
    "Seasonal + Promo": avg_promo_mape,
    "Linear Regression": avg_regression_mape
}


best_model = min(
    model_scores,
    key=model_scores.get
)


# ============================================================
# 11. PRINT MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"Seasonal Baseline      "
    f"MAPE: {avg_seasonal_mape:.2f}%   "
    f"RMSE: {avg_seasonal_rmse:.2f}"
)

print(
    f"Seasonal + Promo       "
    f"MAPE: {avg_promo_mape:.2f}%   "
    f"RMSE: {avg_promo_rmse:.2f}"
)

print(
    f"Linear Regression      "
    f"MAPE: {avg_regression_mape:.2f}%   "
    f"RMSE: {avg_regression_rmse:.2f}"
)

print("-" * 70)

print(
    f"Best model based on MAPE: "
    f"{best_model}"
)

print("=" * 70)


# ============================================================
# 12. FINAL FORECAST + INVENTORY OPTIMIZATION
# ============================================================

fc_rows = []

inventory_rows = []

best_model_mape_by_sku = {}


for sku, g in df.groupby("sku"):

    g = (
        g
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Select best model for this SKU
    # --------------------------------------------------------

    sku_scores = {
        "Seasonal Baseline":
            seasonal_mape[sku],

        "Seasonal + Promo":
            promo_mape[sku],

        "Linear Regression":
            regression_mape[sku]
    }

    sku_best_model = min(
        sku_scores,
        key=sku_scores.get
    )

    best_model_mape_by_sku[sku] = {
        "best_model":
            sku_best_model,

        "mape":
            round(
                sku_scores[
                    sku_best_model
                ],
                2
            )
    }

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    future_dates = pd.date_range(
        g["date"].iloc[-1]
        + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D"
    )

    # --------------------------------------------------------
    # Assume no known future promotions
    # --------------------------------------------------------

    future_promo = np.zeros(
        HORIZON
    )

    # --------------------------------------------------------
    # Fit models using all historical data
    # --------------------------------------------------------

    (
        a,
        b,
        wk,
        mo,
        promo_effect
    ) = seasonal_fit(g)

    regression_model = regression_fit(
        g
    )

    # --------------------------------------------------------
    # Generate seasonal forecast
    # --------------------------------------------------------

    seasonal_forecast = seasonal_predict(
        a,
        b,
        wk,
        mo,
        promo_effect,
        future_dates,
        future_promo,
        len(g)
    )

    # --------------------------------------------------------
    # Generate regression forecast
    # --------------------------------------------------------

    regression_forecast = regression_predict(
        regression_model,
        future_dates,
        future_promo,
        len(g)
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    if sku_best_model in [
        "Seasonal Baseline",
        "Seasonal + Promo"
    ]:

        future_forecast = seasonal_forecast

    else:

        future_forecast = regression_forecast

    # --------------------------------------------------------
    # Save 30-day forecast
    # --------------------------------------------------------

    for d, v in zip(
        future_dates,
        future_forecast
    ):

        fc_rows.append(
            (
                sku,
                d.date().isoformat(),
                round(float(v), 1),
                sku_best_model
            )
        )

    # ========================================================
    # 13. DEMAND VOLATILITY
    # ========================================================

    daily_std = float(
        g["units_sold"]
        .tail(90)
        .std()
    )

    mean_daily = float(
        future_forecast.mean()
    )

    lead_time_demand = (
        mean_daily * LEAD
    )

    # ========================================================
    # 14. SERVICE LEVEL ANALYSIS
    # ========================================================

    review_period = 7

    for service_level, z in SERVICE_LEVELS.items():

        # ----------------------------------------------------
        # Safety stock
        # ----------------------------------------------------

        safety_stock = (
            z
            * daily_std
            * np.sqrt(LEAD)
        )

        # ----------------------------------------------------
        # Reorder point
        # ----------------------------------------------------

        reorder_point = (
            lead_time_demand
            + safety_stock
        )

        # ----------------------------------------------------
        # Order-up-to level
        #
        # Covers:
        # lead time + review period
        # plus safety stock
        # ----------------------------------------------------

        order_up_to = (
            mean_daily
            * (
                LEAD
                + review_period
            )
            + safety_stock
        )

        # ====================================================
        # 15. INVENTORY COST MODEL
        # ====================================================

        # ----------------------------------------------------
        # Cycle inventory
        # ----------------------------------------------------

        cycle_inventory = (
            mean_daily
            * review_period
            / 2
        )

        # ----------------------------------------------------
        # Average inventory
        # ----------------------------------------------------

        average_inventory = (
            cycle_inventory
            + safety_stock
        )

        # ----------------------------------------------------
        # Holding cost
        # ----------------------------------------------------

        holding_cost = (
            average_inventory
            * HOLDING_COST_PER_UNIT_PER_YEAR
        )

        # ----------------------------------------------------
        # Annual demand
        # ----------------------------------------------------

        annual_demand = (
            mean_daily
            * 365
        )

        # ----------------------------------------------------
        # Periodic review ordering
        #
        # One review/order approximately every 7 days.
        # ----------------------------------------------------

        orders_per_year = (
            365
            / review_period
        )

        ordering_cost = (
            orders_per_year
            * ORDERING_COST_PER_ORDER
        )

        # ====================================================
        # 16. EXPECTED SHORTAGE
        # ====================================================
        #
        # Assume lead-time demand follows a Normal
        # distribution.
        #
        # Expected shortage per cycle:
        #
        # sigma_L * [
        #     phi(z)
        #     - z * (1 - Phi(z))
        # ]
        #
        # where:
        # sigma_L = daily_std * sqrt(LEAD)
        #
        # ====================================================

        lead_time_sigma = (
            daily_std
            * np.sqrt(LEAD)
        )

        # Standard normal PDF
        phi_z = (
            1
            / np.sqrt(2 * np.pi)
        ) * np.exp(
            -0.5 * z ** 2
        )

        # Standard normal CDF
        Phi_z = (
            0.5
            * (
                1
                + math.erf(
                    z / np.sqrt(2)
                )
            )
        )

        expected_shortage_per_cycle = (
            lead_time_sigma
            * (
                phi_z
                - z * (
                    1 - Phi_z
                )
            )
        )

        # Number of replenishment cycles per year
        cycles_per_year = (
            365
            / review_period
        )

        expected_annual_shortage = (
            expected_shortage_per_cycle
            * cycles_per_year
        )

        # ----------------------------------------------------
        # Stockout cost
        # ----------------------------------------------------

        stockout_cost = (
            expected_annual_shortage
            * STOCKOUT_COST_PER_UNIT
        )

        # ----------------------------------------------------
        # Total inventory cost
        # ----------------------------------------------------

        total_inventory_cost = (
            holding_cost
            + ordering_cost
            + stockout_cost
        )

        # ====================================================
        # 17. EOQ
        # ====================================================
        #
        # EOQ = sqrt(2DS / H)
        #
        # D = annual demand
        # S = ordering cost per order
        # H = annual holding cost per unit
        #
        # IMPORTANT:
        # EOQ is informational only.
        # It does NOT replace the current 7-day
        # periodic-review replenishment policy.
        # ====================================================

        eoq = np.sqrt(
            (
                2
                * annual_demand
                * ORDERING_COST_PER_ORDER
            )
            / HOLDING_COST_PER_UNIT_PER_YEAR
        )

        eoq_orders_per_year = (
            annual_demand
            / eoq
        )

        eoq_order_interval_days = (
            365
            / eoq_orders_per_year
        )

        # ----------------------------------------------------
        # Save inventory scenario
        # ----------------------------------------------------

        inventory_rows.append(
            (
                sku,
                g["category"].iloc[0],
                service_level,
                mean_daily,
                daily_std,
                annual_demand,
                lead_time_demand,
                safety_stock,
                reorder_point,
                order_up_to,
                average_inventory,
                holding_cost,
                ordering_cost,
                expected_shortage_per_cycle,
                expected_annual_shortage,
                stockout_cost,
                total_inventory_cost,
                eoq,
                eoq_orders_per_year,
                eoq_order_interval_days,
                sku_best_model
            )
        )


# ============================================================
# 18. CREATE FORECAST DATAFRAME
# ============================================================

forecast = pd.DataFrame(
    fc_rows,
    columns=[
        "sku",
        "date",
        "forecast_units",
        "model"
    ]
)


# ============================================================
# 19. CREATE INVENTORY DATAFRAME
# ============================================================

inventory = pd.DataFrame(
    inventory_rows,
    columns=[
        "sku",
        "category",
        "service_level",
        "avg_daily_demand",
        "demand_std",
        "annual_demand",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "order_up_to_level",
        "average_inventory",
        "holding_cost",
        "ordering_cost",
        "expected_shortage_per_cycle",
        "expected_annual_shortage",
        "stockout_cost",
        "total_inventory_cost",
        "eoq",
        "eoq_orders_per_year",
        "eoq_order_interval_days",
        "model"
    ]
)


# ============================================================
# 20. SAVE FORECAST
# ============================================================

forecast.to_csv(
    BASE / "forecast_30d.csv",
    index=False
)


# ============================================================
# 21. SAVE INVENTORY OPTIMIZATION
# ============================================================

inventory.to_csv(
    BASE / "inventory_optimization.csv",
    index=False
)


# ============================================================
# 22. 95% SERVICE LEVEL INVENTORY RECOMMENDATIONS
# ============================================================

inventory_95 = inventory[
    inventory["service_level"] == "95%"
].copy()

inventory_95.to_csv(
    BASE / "inventory_recommendations.csv",
    index=False
)


# ============================================================
# 23. SERVICE LEVEL SUMMARY
# ============================================================

service_summary = (
    inventory
    .groupby("service_level")
    .agg(
        total_safety_stock=(
            "safety_stock",
            "sum"
        ),

        total_reorder_point=(
            "reorder_point",
            "sum"
        ),

        total_average_inventory=(
            "average_inventory",
            "sum"
        ),

        total_holding_cost=(
            "holding_cost",
            "sum"
        ),

        total_ordering_cost=(
            "ordering_cost",
            "sum"
        ),

        total_stockout_cost=(
            "stockout_cost",
            "sum"
        ),

        total_inventory_cost=(
            "total_inventory_cost",
            "sum"
        )
    )
    .reset_index()
)


service_summary.to_csv(
    BASE / "service_level_analysis.csv",
    index=False
)


# ============================================================
# 24. MODEL COMPARISON TABLE
# ============================================================

comparison_rows = []

for sku in sorted(
    df["sku"].unique()
):

    comparison_rows.append(
        (
            sku,
            seasonal_mape[sku],
            promo_mape[sku],
            regression_mape[sku]
        )
    )


comparison = pd.DataFrame(
    comparison_rows,
    columns=[
        "sku",
        "seasonal_mape",
        "promo_mape",
        "regression_mape"
    ]
)


comparison.to_csv(
    BASE / "model_comparison.csv",
    index=False
)


# ============================================================
# 25. MODEL COMPARISON CHART
# ============================================================

plt.figure(
    figsize=(8, 4)
)

model_names = [
    "Seasonal",
    "Seasonal + Promo",
    "Linear Regression"
]

model_values = [
    avg_seasonal_mape,
    avg_promo_mape,
    avg_regression_mape
]

plt.bar(
    model_names,
    model_values
)

plt.ylabel(
    "Average MAPE (%)"
)

plt.title(
    "Demand Forecasting Model Comparison"
)

plt.tight_layout()

plt.savefig(
    BASE / "model_comparison.png"
)

plt.close()


# ============================================================
# 26. MAPE BY SKU
# ============================================================

plt.figure(
    figsize=(9, 4)
)

x = np.arange(
    len(comparison)
)

width = 0.25

plt.bar(
    x - width,
    comparison["seasonal_mape"],
    width,
    label="Seasonal"
)

plt.bar(
    x,
    comparison["promo_mape"],
    width,
    label="Seasonal + Promo"
)

plt.bar(
    x + width,
    comparison["regression_mape"],
    width,
    label="Linear Regression"
)

plt.xticks(
    x,
    comparison["sku"]
)

plt.ylabel(
    "MAPE (%)"
)

plt.title(
    "Forecasting Error by SKU"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    BASE / "model_mape_by_sku.png"
)

plt.close()


# ============================================================
# 27. SERVICE LEVEL VS INVENTORY
# ============================================================

plt.figure(
    figsize=(8, 4)
)

plt.plot(
    service_summary["service_level"],
    service_summary["total_average_inventory"],
    marker="o"
)

plt.xlabel(
    "Service Level"
)

plt.ylabel(
    "Total Average Inventory"
)

plt.title(
    "Service Level vs Average Inventory"
)

plt.tight_layout()

plt.savefig(
    BASE / "service_level_inventory.png"
)

plt.close()


# ============================================================
# 28. SERVICE LEVEL VS COST
# ============================================================

plt.figure(
    figsize=(8, 4)
)

plt.plot(
    service_summary["service_level"],
    service_summary["total_inventory_cost"],
    marker="o"
)

plt.xlabel(
    "Service Level"
)

plt.ylabel(
    "Estimated Inventory Cost"
)

plt.title(
    "Service Level vs Estimated Inventory Cost"
)

plt.tight_layout()

plt.savefig(
    BASE / "service_level_cost.png"
)

plt.close()


# ============================================================
# 29. REORDER POINT CHART
# ============================================================

ax = (
    inventory_95
    .set_index("sku")[
        [
            "lead_time_demand",
            "safety_stock"
        ]
    ]
    .plot(
        kind="bar",
        stacked=True,
        figsize=(8, 4)
    )
)

ax.set_ylabel(
    "Units"
)

ax.set_title(
    "95% Service Level: Reorder Point"
)

plt.tight_layout()

plt.savefig(
    BASE / "reorder_points.png"
)

plt.close()


# ============================================================
# 30. FORECAST VS ACTUAL
# ============================================================

tot = (
    df
    .groupby("date")["units_sold"]
    .sum()
)

total_df = tot.reset_index()

t_total = np.arange(
    len(total_df)
)

y_total = (
    total_df["units_sold"]
    .values
    .astype(float)
)

# Simple trend
b_total, a_total = np.polyfit(
    t_total,
    y_total,
    1
)

detr_total = (
    y_total
    - (
        a_total
        + b_total * t_total
    )
)

# Weekly effect
wk_total = (
    pd.Series(detr_total)
    .groupby(
        total_df["date"]
        .dt.dayofweek
        .values
    )
    .mean()
)

# Remove weekly effect
weekly_total_values = (
    wk_total
    .reindex(
        total_df["date"]
        .dt.dayofweek
        .values
    )
    .values
)

detr_total_2 = (
    detr_total
    - weekly_total_values
)

# Monthly effect
mo_total = (
    pd.Series(detr_total_2)
    .groupby(
        total_df["date"]
        .dt.month
        .values
    )
    .mean()
)

# Future dates
future_total_dates = pd.date_range(
    tot.index[-1]
    + pd.Timedelta(days=1),
    periods=HORIZON,
    freq="D"
)

future_total_t = np.arange(
    len(tot),
    len(tot) + HORIZON
)

future_total_forecast = (
    a_total
    + b_total * future_total_t
    + wk_total.reindex(
        future_total_dates.dayofweek
    ).values
    + mo_total.reindex(
        future_total_dates.month
    ).values
)

future_total_forecast = np.clip(
    future_total_forecast,
    0,
    None
)


plt.figure(
    figsize=(8, 3.2)
)

plt.plot(
    tot.index[-120:],
    tot.values[-120:],
    label="Actual"
)

plt.plot(
    future_total_dates,
    future_total_forecast,
    "--",
    label="Forecast (30d)"
)

plt.title(
    "Total Daily Demand: Actual + Forecast"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    BASE / "forecast_vs_actual.png"
)

plt.close()


# ============================================================
# 31. WEEKLY SEASONALITY
# ============================================================

weekly = (
    df
    .groupby(
        df["date"].dt.dayofweek
    )["units_sold"]
    .mean()
)

ax = weekly.plot(
    kind="bar",
    figsize=(8, 4)
)

ax.set_xticklabels(
    [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun"
    ],
    rotation=0
)

ax.set_ylabel(
    "Average Units"
)

ax.set_title(
    "Weekly Demand Seasonality"
)

plt.tight_layout()

plt.savefig(
    BASE / "weekly_seasonality.png"
)

plt.close()


# ============================================================
# 32. FINAL RESULTS JSON
# ============================================================

results = {

    "skus":
        int(
            df["sku"].nunique()
        ),

    "days":
        int(
            df["date"].nunique()
        ),

    "model_comparison": {

        "Seasonal Baseline":
            round(
                float(avg_seasonal_mape),
                2
            ),

        "Seasonal + Promo":
            round(
                float(avg_promo_mape),
                2
            ),

        "Linear Regression":
            round(
                float(avg_regression_mape),
                2
            )
    },

    "model_rmse": {

        "Seasonal Baseline":
            round(
                float(avg_seasonal_rmse),
                2
            ),

        "Seasonal + Promo":
            round(
                float(avg_promo_rmse),
                2
            ),

        "Linear Regression":
            round(
                float(avg_regression_rmse),
                2
            )
    },

    "best_overall_model":
        best_model,

    "best_model_by_sku":
        best_model_mape_by_sku,

    "forecast_horizon_days":
        HORIZON,

    "lead_time_days":
        LEAD,

    "service_level":
        "95%",

    "service_levels_tested":
        list(
            SERVICE_LEVELS.keys()
        ),

    "cost_assumptions": {

        "holding_cost_per_unit_per_year":
            HOLDING_COST_PER_UNIT_PER_YEAR,

        "stockout_cost_per_unit":
            STOCKOUT_COST_PER_UNIT,

        "ordering_cost_per_order":
            ORDERING_COST_PER_ORDER
    },

    "total_reorder_units_95":
        int(
            inventory_95[
                "reorder_point"
            ].sum()
        )
}


(
    BASE / "results.json"
).write_text(
    json.dumps(
        results,
        indent=2
    )
)


# ============================================================
# 33. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"Seasonal Baseline      "
    f"MAPE: {avg_seasonal_mape:.2f}%   "
    f"RMSE: {avg_seasonal_rmse:.2f}"
)

print(
    f"Seasonal + Promo       "
    f"MAPE: {avg_promo_mape:.2f}%   "
    f"RMSE: {avg_promo_rmse:.2f}"
)

print(
    f"Linear Regression      "
    f"MAPE: {avg_regression_mape:.2f}%   "
    f"RMSE: {avg_regression_rmse:.2f}"
)

print("-" * 70)

print(
    f"Best model based on MAPE: "
    f"{best_model}"
)

print("=" * 70)


# ============================================================
# SERVICE LEVEL ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("SERVICE LEVEL ANALYSIS")
print("=" * 70)

print(
    service_summary.to_string(
        index=False
    )
)

print("=" * 70)


# ============================================================
# 95% SERVICE LEVEL INVENTORY PLAN
# ============================================================

print("\n")
print("=" * 70)
print("95% SERVICE LEVEL INVENTORY PLAN")
print("=" * 70)

print(
    inventory_95[
        [
            "sku",
            "category",
            "avg_daily_demand",
            "demand_std",
            "lead_time_demand",
            "safety_stock",
            "reorder_point",
            "order_up_to_level",
            "average_inventory",
            "holding_cost",
            "ordering_cost",
            "stockout_cost",
            "total_inventory_cost",
            "eoq",
            "eoq_order_interval_days",
            "model"
        ]
    ].to_string(
        index=False
    )
)

print("=" * 70)


# ============================================================
# GENERATED FILES
# ============================================================

print("\n")
print("Generated files:")

print("- forecast_30d.csv")
print("- inventory_optimization.csv")
print("- inventory_recommendations.csv")
print("- service_level_analysis.csv")
print("- model_comparison.csv")
print("- model_comparison.png")
print("- model_mape_by_sku.png")
print("- service_level_inventory.png")
print("- service_level_cost.png")
print("- reorder_points.png")
print("- forecast_vs_actual.png")
print("- weekly_seasonality.png")
print("- results.json")

