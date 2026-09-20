"""
forecast.py - Demand Forecasting & Inventory Optimization

Per SKU:
1. Fit trend + weekly + monthly seasonality + promotion effect
2. Backtest on a 60-day holdout using MAPE and RMSE
3. Forecast the next 30 days
4. Compute safety stock, reorder point, and order-up-to level
5. Export forecasts, inventory recommendations, results, and charts
"""

import pathlib
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD DATA
# ============================================================

BASE = pathlib.Path(__file__).resolve().parent

df = (
    pd.read_csv(
        BASE / "sales.csv",
        parse_dates=["date"]
    )
    .sort_values(["sku", "date"])
)


# ============================================================
# 2. PARAMETERS
# ============================================================

HORIZON = 30       # Forecast next 30 days
HOLDOUT = 60       # Use last 60 days for backtesting
LEAD = 7           # Supplier lead time = 7 days
Z = 1.65           # 95% service level


# ============================================================
# 3. MODEL FITTING
# ============================================================

def seasonal_fit(train):
    """
    Learn:
    - Linear trend
    - Weekly seasonality
    - Monthly seasonality
    - Promotion effect
    """

    # Time index
    t = np.arange(len(train))

    # Actual demand
    y = train["units_sold"].values.astype(float)

    # --------------------------------------------------------
    # Linear trend
    # --------------------------------------------------------

    b, a = np.polyfit(t, y, 1)

    # Remove trend from demand
    detr = y - (a + b * t)

    # --------------------------------------------------------
    # Weekly seasonality
    # --------------------------------------------------------

    wk = (
        pd.Series(detr)
        .groupby(train["date"].dt.dayofweek.values)
        .mean()
    )

    # --------------------------------------------------------
    # Monthly seasonality
    # --------------------------------------------------------

    weekly_effect = wk.reindex(
        train["date"].dt.dayofweek.values
    ).values

    mo = (
        pd.Series(detr - weekly_effect)
        .groupby(train["date"].dt.month.values)
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

    return a, b, wk, mo, promo_effect, len(train)


# ============================================================
# 4. PREDICTION
# ============================================================

def predict(a, b, wk, mo, promo_effect, dates, promo, t0):
    """
    Generate demand predictions using:

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

    # Trend component
    base = a + b * t

    # Calendar features
    dow = dates.dt.dayofweek.values
    month = dates.dt.month.values

    # Promotion flag
    promo = np.asarray(promo)

    # Final prediction
    prediction = (
        base
        + wk.reindex(dow).values
        + mo.reindex(month).values
        + promo * promo_effect
    )

    # Demand cannot be negative
    return np.clip(
        prediction,
        0,
        None
    )


# ============================================================
# 5. FORECAST EACH SKU
# ============================================================

fc_rows = []
inv_rows = []
mapes = {}

for sku, g in df.groupby("sku"):

    g = g.reset_index(drop=True)

    # --------------------------------------------------------
    # Split data into training and test sets
    # --------------------------------------------------------

    train = g.iloc[:-HOLDOUT]
    test = g.iloc[-HOLDOUT:]

    # --------------------------------------------------------
    # Fit model using training data
    # --------------------------------------------------------

    (
        a,
        b,
        wk,
        mo,
        promo_effect,
        n
    ) = seasonal_fit(train)

    # --------------------------------------------------------
    # Backtest prediction
    # --------------------------------------------------------

    pred = predict(
        a,
        b,
        wk,
        mo,
        promo_effect,
        test["date"],
        test["promo"],
        len(train)
    )

    actual = test["units_sold"].values

    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    mape = float(
        np.mean(
            np.abs(
                (actual - pred)
                / np.clip(actual, 1, None)
            )
        ) * 100
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(
                (actual - pred) ** 2
            )
        )
    )

    mapes[sku] = round(mape, 1)

    # --------------------------------------------------------
    # Refit model using the full historical dataset
    # --------------------------------------------------------

    (
        a,
        b,
        wk,
        mo,
        promo_effect,
        n
    ) = seasonal_fit(g)

    # --------------------------------------------------------
    # Future dates
    # --------------------------------------------------------

    fut = pd.date_range(
        g["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D"
    )

    # --------------------------------------------------------
    # Temporary assumption:
    # no promotions in the next 30 days
    # --------------------------------------------------------

    future_promo = np.zeros(HORIZON)

    # --------------------------------------------------------
    # Forecast next 30 days
    # --------------------------------------------------------

    fut_pred = predict(
        a,
        b,
        wk,
        mo,
        promo_effect,
        fut,
        future_promo,
        len(g)
    )

    # Save forecasts
    for d, v in zip(fut, fut_pred):

        fc_rows.append(
            (
                sku,
                d.date().isoformat(),
                round(float(v), 1)
            )
        )

    # ========================================================
    # 6. INVENTORY OPTIMIZATION
    # ========================================================

    # Recent demand volatility
    daily_std = float(
        g["units_sold"].tail(90).std()
    )

    # Average forecasted daily demand
    mean_daily = float(
        fut_pred.mean()
    )

    # Expected demand during lead time
    lt_demand = mean_daily * LEAD

    # Safety stock
    safety = (
        Z
        * daily_std
        * np.sqrt(LEAD)
    )

    # Reorder point
    reorder = (
        lt_demand
        + safety
    )

    # Order-up-to level
    order_up_to = (
        mean_daily
        * (LEAD + 7)
        + safety
    )

    inv_rows.append(
        (
            sku,
            g["category"].iloc[0],
            round(mean_daily, 1),
            round(daily_std, 1),
            round(lt_demand, 0),
            round(safety, 0),
            round(reorder, 0),
            round(order_up_to, 0)
        )
    )


# ============================================================
# 7. CREATE OUTPUT DATAFRAMES
# ============================================================

fc = pd.DataFrame(
    fc_rows,
    columns=[
        "sku",
        "date",
        "forecast_units"
    ]
)

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
        "order_up_to_level"
    ]
)


# ============================================================
# 8. SAVE OUTPUT FILES
# ============================================================

fc.to_csv(
    BASE / "forecast_30d.csv",
    index=False
)

inv.to_csv(
    BASE / "inventory_recommendations.csv",
    index=False
)


# ============================================================
# 9. CHARTS
# ============================================================

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 10
    }
)


# ------------------------------------------------------------
# Chart 1: Forecast vs Actual
# ------------------------------------------------------------

tot = (
    df.groupby("date")["units_sold"]
    .sum()
)

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

# Fit simple trend
b_total, a_total = np.polyfit(
    t_total,
    y_total,
    1
)

detr_total = (
    y_total
    - (a_total + b_total * t_total)
)

# Weekly effect
wk_total = (
    pd.Series(detr_total)
    .groupby(
        total_df["date"].dt.dayofweek.values
    )
    .mean()
)

# Monthly effect
weekly_total = (
    wk_total
    .reindex(
        total_df["date"].dt.dayofweek.values
    )
    .values
)

mo_total = (
    pd.Series(
        detr_total - weekly_total
    )
    .groupby(
        total_df["date"].dt.month.values
    )
    .mean()
)

# Future dates
futd = pd.date_range(
    tot.index[-1] + pd.Timedelta(days=1),
    periods=HORIZON
)

t_future = np.arange(
    len(tot),
    len(tot) + HORIZON
)

total_forecast = (
    a_total
    + b_total * t_future
    + wk_total.reindex(
        futd.dayofweek
    ).values
    + mo_total.reindex(
        futd.month
    ).values
)

total_forecast = np.clip(
    total_forecast,
    0,
    None
)

plt.figure(
    figsize=(8, 3.2)
)

plt.plot(
    tot.index[-120:],
    tot.values[-120:],
    label="Actual",
    color="#34495e"
)

plt.plot(
    futd,
    total_forecast,
    "--",
    label="Forecast (30d)",
    color="#e67e22"
)

plt.title(
    "Total daily demand: recent actual + 30-day forecast"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    BASE / "forecast_vs_actual.png"
)

plt.close()


# ------------------------------------------------------------
# Chart 2: Weekly Seasonality
# ------------------------------------------------------------

wkf = (
    df.assign(
        dow=df["date"].dt.day_name().str[:3]
    )
    .groupby(
        df["date"].dt.dayofweek
    )["units_sold"]
    .mean()
)

ax = wkf.plot(
    kind="bar",
    color="#2980b9"
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
    "Avg units"
)

ax.set_title(
    "Weekly demand seasonality"
)

plt.tight_layout()

plt.savefig(
    BASE / "weekly_seasonality.png"
)

plt.close()


# ------------------------------------------------------------
# Chart 3: MAPE by SKU
# ------------------------------------------------------------

ax = pd.Series(mapes).plot(
    kind="bar",
    color="#16a085"
)

ax.set_ylabel(
    "MAPE %"
)

ax.set_title(
    "Backtest forecast error by SKU (lower is better)"
)

plt.tight_layout()

plt.savefig(
    BASE / "mape_by_sku.png"
)

plt.close()


# ------------------------------------------------------------
# Chart 4: Reorder Point vs Safety Stock
# ------------------------------------------------------------

axx = (
    inv.set_index("sku")
    [
        [
            "lead_time_demand",
            "safety_stock"
        ]
    ]
    .plot(
        kind="bar",
        stacked=True,
        color=[
            "#3498db",
            "#e74c3c"
        ]
    )
)

axx.set_ylabel(
    "Units"
)

axx.set_title(
    "Reorder point = lead-time demand + safety stock"
)

plt.tight_layout()

plt.savefig(
    BASE / "reorder_points.png"
)

plt.close()


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results = {
    "skus": int(
        df.sku.nunique()
    ),

    "days": int(
        df.date.nunique()
    ),

    "backtest_mape_by_sku": mapes,

    "avg_mape": round(
        float(
            np.mean(
                list(mapes.values())
            )
        ),
        1
    ),

    "forecast_horizon_days": HORIZON,

    "lead_time_days": LEAD,

    "service_level": "95%",

    "total_reorder_units": int(
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
# 11. PRINT RESULTS
# ============================================================

print(
    json.dumps(
        results,
        indent=2
    )
)

print(
    inv.to_string(
        index=False
    )
)