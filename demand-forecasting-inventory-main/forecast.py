
"""
forecast.py
AI-Driven Supply Chain Demand Forecasting & Inventory Optimization

Day 2:
1. Seasonal forecasting model
2. Seasonal + Promotion model
3. Linear Regression model
4. Backtest and compare MAPE
5. Select the best model
6. Forecast next 30 days
7. Calculate safety stock, reorder point, and order-up-to level
"""

import pathlib
import json

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
Z = 1.65   # 95% service level


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    BASE / "sales.csv",
    parse_dates=["date"]
)

df = df.sort_values(["sku", "date"]).reset_index(drop=True)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def create_features(data):
    """
    Create features for Linear Regression.

    Features:
    - trend
    - day of week
    - month
    - promotion
    """

    data = data.copy()

    # Sequential time index
    data["t"] = np.arange(len(data))

    # Day of week: Monday=0 ... Sunday=6
    data["dow"] = data["date"].dt.dayofweek

    # Month: 1 ... 12
    data["month"] = data["date"].dt.month

    return data


# ============================================================
# 4. ORIGINAL SEASONAL MODEL
# ============================================================

def seasonal_fit(train):
    """
    Fit:
        trend
        weekly seasonality
        monthly seasonality
        promotion effect
    """

    t = np.arange(len(train))

    y = train["units_sold"].values.astype(float)

    # Linear trend
    b, a = np.polyfit(t, y, 1)

    # Remove trend
    detr = y - (a + b * t)

    # Weekly effect
    wk = (
        pd.Series(detr)
        .groupby(train["date"].dt.dayofweek.values)
        .mean()
    )

    # Remove weekly effect
    weekly_values = wk.reindex(
        train["date"].dt.dayofweek.values
    ).values

    detr2 = detr - weekly_values

    # Monthly effect
    mo = (
        pd.Series(detr2)
        .groupby(train["date"].dt.month.values)
        .mean()
    )

    # Promotion effect
    promo_mean = train.loc[
        train["promo"] == 1,
        "units_sold"
    ].mean()

    non_promo_mean = train.loc[
        train["promo"] == 0,
        "units_sold"
    ].mean()

    promo_effect = promo_mean - non_promo_mean

    return a, b, wk, mo, promo_effect


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
    Generate predictions using the seasonal model.
    """

    dates = pd.Series(
        pd.to_datetime(dates)
    ).reset_index(drop=True)

    t = np.arange(
        t0,
        t0 + len(dates)
    )

    base = a + b * t

    dow = dates.dt.dayofweek.values
    month = dates.dt.month.values

    weekly_effect = wk.reindex(dow).values
    monthly_effect = mo.reindex(month).values

    prediction = (
        base
        + weekly_effect
        + monthly_effect
        + promo * promo_effect
    )

    return np.clip(
        prediction,
        0,
        None
    )


# ============================================================
# 5. LINEAR REGRESSION MODEL
# ============================================================

def regression_fit(train):
    """
    Train a Linear Regression model.

    Features:
        t
        day of week
        month
        promo
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


def regression_predict(model, dates, promo, t0):
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

    future["dow"] = dates.dt.dayofweek.values

    future["month"] = dates.dt.month.values

    future["promo"] = np.asarray(promo)

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
# 6. EVALUATION METRICS
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
# 7. BACKTEST ALL MODELS
# ============================================================

seasonal_mape = {}
promo_mape = {}
regression_mape = {}

seasonal_rmse = {}
promo_rmse = {}
regression_rmse = {}


# We will use the same data split for every model.

for sku, g in df.groupby("sku"):

    g = g.reset_index(drop=True)

    train = g.iloc[:-HOLDOUT]

    test = g.iloc[-HOLDOUT:]

    # --------------------------------------------------------
    # MODEL 1
    # Seasonal Baseline
    # --------------------------------------------------------

    a, b, wk, mo, promo_effect = seasonal_fit(train)

    # Ignore promotion
    no_promo = np.zeros(len(test))

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

    # --------------------------------------------------------
    # MODEL 2
    # Seasonal + Promotion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MODEL 3
    # Linear Regression
    # --------------------------------------------------------

    regression_model = regression_fit(train)

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
# 8. MODEL COMPARISON
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
# 9. PRINT MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

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

print("-" * 65)

print(
    f"Best model based on MAPE: {best_model}"
)

print("=" * 65)
print("\n")


# ============================================================
# 10. FINAL FORECAST + INVENTORY OPTIMIZATION
# ============================================================

fc_rows = []

inv_rows = []

best_model_mape_by_sku = {}


for sku, g in df.groupby("sku"):

    g = g.reset_index(drop=True)

    # --------------------------------------------------------
    # Calculate which model performed best for this SKU
    # --------------------------------------------------------

    sku_scores = {
        "Seasonal Baseline": seasonal_mape[sku],
        "Seasonal + Promo": promo_mape[sku],
        "Linear Regression": regression_mape[sku]
    }

    sku_best_model = min(
        sku_scores,
        key=sku_scores.get
    )

    best_model_mape_by_sku[sku] = {
        "best_model": sku_best_model,
        "mape": round(
            sku_scores[sku_best_model],
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
    # Fit final models using ALL historical data
    # --------------------------------------------------------

    a, b, wk, mo, promo_effect = seasonal_fit(g)

    regression_model = regression_fit(g)

    # --------------------------------------------------------
    # Generate forecasts from both models
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

    regression_forecast = regression_predict(
        regression_model,
        future_dates,
        future_promo,
        len(g)
    )

    # --------------------------------------------------------
    # Select the best model for this SKU
    # --------------------------------------------------------

    if sku_best_model == "Seasonal Baseline":

        future_forecast = seasonal_predict(
            a,
            b,
            wk,
            mo,
            promo_effect,
            future_dates,
            future_promo,
            len(g)
        )

    elif sku_best_model == "Seasonal + Promo":

        # There are no known future promotions,
        # so future promo = 0.

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
    # INVENTORY OPTIMIZATION
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

    safety_stock = (
        Z
        * daily_std
        * np.sqrt(LEAD)
    )

    reorder_point = (
        lead_time_demand
        + safety_stock
    )

    review_period = 7

    order_up_to = (
        mean_daily
        * (LEAD + review_period)
        + safety_stock
    )

    inv_rows.append(
        (
            sku,
            g["category"].iloc[0],
            round(mean_daily, 1),
            round(daily_std, 1),
            round(lead_time_demand, 0),
            round(safety_stock, 0),
            round(reorder_point, 0),
            round(order_up_to, 0),
            sku_best_model
        )
    )


# ============================================================
# 11. SAVE FORECAST DATA
# ============================================================

fc = pd.DataFrame(
    fc_rows,
    columns=[
        "sku",
        "date",
        "forecast_units",
        "model"
    ]
)

fc.to_csv(
    BASE / "forecast_30d.csv",
    index=False
)


# ============================================================
# 12. SAVE INVENTORY RECOMMENDATIONS
# ============================================================

inv = pd.DataFrame(
    inv_rows,
    columns=[
        "sku",
        "category",
        "avg_daily_demand",
        "demand_std",
        "lead_time_demand",
        "safety_stock",
        "reorder_point",
        "order_up_to_level",
        "model"
    ]
)

inv.to_csv(
    BASE / "inventory_recommendations.csv",
    index=False
)


# ============================================================
# 13. MODEL COMPARISON TABLE
# ============================================================

comparison_rows = []

for sku in sorted(df["sku"].unique()):

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
# 14. MODEL COMPARISON CHART
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

plt.ylabel("Average MAPE (%)")

plt.title(
    "Demand Forecasting Model Comparison"
)

plt.tight_layout()

plt.savefig(
    BASE / "model_comparison.png"
)

plt.close()


# ============================================================
# 15. MAPE BY SKU
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
# 16. FORECAST VS ACTUAL
# ============================================================

tot = (
    df.groupby("date")["units_sold"]
    .sum()
)

# Simple total-demand trend + weekly + monthly model
total_df = (
    tot.reset_index()
)

t_total = np.arange(
    len(total_df)
)

y_total = (
    total_df["units_sold"]
    .values
    .astype(float)
)

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

wk_total = (
    pd.Series(detr_total)
    .groupby(
        total_df["date"]
        .dt.dayofweek
        .values
    )
    .mean()
)

mo_total = (
    pd.Series(
        detr_total
        - wk_total.reindex(
            total_df["date"]
            .dt.dayofweek
            .values
        ).values
    )
    .groupby(
        total_df["date"]
        .dt.month
        .values
    )
    .mean()
)

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
# 17. WEEKLY SEASONALITY
# ============================================================

weekly = (
    df.assign(
        dow=df["date"]
        .dt.day_name()
        .str[:3]
    )
    .groupby(
        df["date"]
        .dt.dayofweek
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
# 18. REORDER POINT CHART
# ============================================================

ax = (
    inv.set_index("sku")[
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
    "Reorder Point = Lead-Time Demand + Safety Stock"
)

plt.tight_layout()

plt.savefig(
    BASE / "reorder_points.png"
)

plt.close()


# ============================================================
# 19. SAVE FINAL RESULTS
# ============================================================

results = {

    "skus": int(
        df["sku"].nunique()
    ),

    "days": int(
        df["date"].nunique()
    ),

    "model_comparison": {

        "Seasonal Baseline": round(
            float(avg_seasonal_mape),
            2
        ),

        "Seasonal + Promo": round(
            float(avg_promo_mape),
            2
        ),

        "Linear Regression": round(
            float(avg_regression_mape),
            2
        )
    },

    "best_overall_model": best_model,

    "best_model_by_sku":
        best_model_mape_by_sku,

    "forecast_horizon_days":
        HORIZON,

    "lead_time_days":
        LEAD,

    "service_level":
        "95%",

    "total_reorder_units":
        int(
            inv["reorder_point"].sum()
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
# 20. FINAL OUTPUT
# ============================================================

print(
    json.dumps(
        results,
        indent=2
    )
)

print("\nInventory Recommendations:\n")

print(
    inv.to_string(
        index=False
    )
)

print("\n")

print(
    "Generated files:"
)

print(
    "- forecast_30d.csv"
)

print(
    "- inventory_recommendations.csv"
)

print(
    "- model_comparison.csv"
)

print(
    "- model_comparison.png"
)

print(
    "- model_mape_by_sku.png"
)

print(
    "- forecast_vs_actual.png"
)

print(
    "- weekly_seasonality.png"
)

print(
    "- reorder_points.png"
)

print(
    "- results.json"
)

