from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 基础配置
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "sales.csv"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FORECAST_HORIZON = 30
BACKTEST_DAYS = 60

LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7

# 我们比较三种服务水平
SERVICE_LEVELS = {
    "90%": 1.28,
    "95%": 1.65,
    "99%": 2.33,
}

# ============================================================
# 成本假设
# ============================================================
# 注意：
# 这些是项目中的“示例成本参数”，不是某个真实公司的财务数据。

HOLDING_COST_PER_UNIT_PER_YEAR = 5.00
STOCKOUT_COST_PER_UNIT = 5.00
ORDERING_COST_PER_ORDER = 100.00


# ============================================================
# 2. 数据读取
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = df.sort_values(
    ["sku", "date"]
).reset_index(drop=True)


# ============================================================
# 3. MAPE / RMSE
# ============================================================

def mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def rmse(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )


# ============================================================
# 4. Seasonal Baseline / Seasonal + Promo
# ============================================================

def seasonal_fit(train_df, use_promo=False):
    data = train_df.copy()

    data["dow"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    # ----------------------------
    # 基础趋势
    # ----------------------------

    x = np.arange(len(data))

    if len(data) >= 2:
        trend_coef = np.polyfit(
            x,
            data["units_sold"],
            1
        )
    else:
        trend_coef = [
            0,
            data["units_sold"].mean()
        ]

    trend = np.polyval(
        trend_coef,
        x
    )

    residual = (
        data["units_sold"]
        - trend
    )

    # ----------------------------
    # 星期季节性
    # ----------------------------

    dow_factor = (
        residual.groupby(data["dow"])
        .mean()
    )

    # ----------------------------
    # 月份季节性
    # ----------------------------

    month_factor = (
        residual.groupby(data["month"])
        .mean()
    )

    # ----------------------------
    # Promotion effect
    # ----------------------------

    promo_effect = 0.0

    if use_promo and "promo" in data.columns:

        promo_days = data.loc[
            data["promo"] == 1,
            "units_sold"
        ]

        non_promo_days = data.loc[
            data["promo"] == 0,
            "units_sold"
        ]

        if (
            len(promo_days) > 0
            and len(non_promo_days) > 0
        ):

            promo_effect = (
                promo_days.mean()
                - non_promo_days.mean()
            )

    return (
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
    )


def seasonal_predict(
    dates,
    trend_coef,
    dow_factor,
    month_factor,
    promo_effect=0.0,
    promo=None
):

    dates = pd.Series(
        pd.to_datetime(dates)
    )

    x_future = np.arange(
        len(dates)
    )

    trend = np.polyval(
        trend_coef,
        x_future
    )

    dow = dates.dt.dayofweek
    month = dates.dt.month

    dow_component = dow.map(
        dow_factor
    ).fillna(0).values

    month_component = month.map(
        month_factor
    ).fillna(0).values

    prediction = (
        trend
        + dow_component
        + month_component
    )

    if promo is not None:

        promo = np.asarray(
            promo
        )

        prediction = (
            prediction
            + promo * promo_effect
        )

    return np.maximum(
        prediction,
        0
    )


# ============================================================
# 5. Linear Regression
# ============================================================

def regression_fit(train_df):

    data = train_df.copy()

    data["t"] = np.arange(
        len(data)
    )

    data["dow"] = (
        data["date"].dt.dayofweek
    )

    data["month"] = (
        data["date"].dt.month
    )

    features = [
        "t",
        "dow",
        "month",
        "promo",
    ]

    X = data[features].values

    y = data["units_sold"].values

    X_design = np.column_stack(
        [
            np.ones(len(X)),
            X
        ]
    )

    coef = np.linalg.lstsq(
        X_design,
        y,
        rcond=None
    )[0]

    return coef


def regression_predict(
    dates,
    coef,
    promo=None,
    start_t=0
):

    dates = pd.Series(
        pd.to_datetime(dates)
    )

    if promo is None:
        promo = np.zeros(
            len(dates)
        )

    t = (
        np.arange(len(dates))
        + start_t
    )

    dow = (
        dates.dt.dayofweek.values
    )

    month = (
        dates.dt.month.values
    )

    X = np.column_stack(
        [
            np.ones(len(dates)),
            t,
            dow,
            month,
            promo,
        ]
    )

    prediction = X @ coef

    return np.maximum(
        prediction,
        0
    )


# ============================================================
# 6. 模型回测
# ============================================================

model_results = []

sku_model_predictions = {}


for sku in sorted(
    df["sku"].unique()
):

    sku_df = df[
        df["sku"] == sku
    ].sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    train = sku_df.iloc[
        :-BACKTEST_DAYS
    ].copy()

    test = sku_df.iloc[
        -BACKTEST_DAYS:
    ].copy()

    # --------------------------------------------------------
    # Model 1: Seasonal Baseline
    # --------------------------------------------------------

    (
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
    ) = seasonal_fit(
        train,
        use_promo=False
    )

    pred_seasonal = seasonal_predict(
        test["date"],
        trend_coef,
        dow_factor,
        month_factor
    )

    seasonal_mape = mape(
        test["units_sold"],
        pred_seasonal
    )

    seasonal_rmse = rmse(
        test["units_sold"],
        pred_seasonal
    )

    # --------------------------------------------------------
    # Model 2: Seasonal + Promo
    # --------------------------------------------------------

    (
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
    ) = seasonal_fit(
        train,
        use_promo=True
    )

    pred_promo = seasonal_predict(
        test["date"],
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
        test["promo"]
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
    # Model 3: Linear Regression
    # --------------------------------------------------------

    regression_coef = regression_fit(
        train
    )

    pred_regression = regression_predict(
        test["date"],
        regression_coef,
        test["promo"].values,
        start_t=len(train)
    )

    regression_mape = mape(
        test["units_sold"],
        pred_regression
    )

    regression_rmse = rmse(
        test["units_sold"],
        pred_regression
    )

    # --------------------------------------------------------
    # 保存模型结果
    # --------------------------------------------------------

    models = {
        "Seasonal Baseline": (
            seasonal_mape,
            seasonal_rmse,
            pred_seasonal
        ),

        "Seasonal + Promo": (
            promo_mape,
            promo_rmse,
            pred_promo
        ),

        "Linear Regression": (
            regression_mape,
            regression_rmse,
            pred_regression
        ),
    }

    best_model = min(
        models,
        key=lambda x: models[x][0]
    )

    sku_model_predictions[sku] = {
        "best_model": best_model,
        "predictions": models,
    }

    for model_name, (
        model_mape,
        model_rmse,
        _
    ) in models.items():

        model_results.append(
            {
                "sku": sku,
                "model": model_name,
                "mape": model_mape,
                "rmse": model_rmse,
                "is_best": (
                    model_name
                    == best_model
                ),
            }
        )


model_results_df = pd.DataFrame(
    model_results
)

model_results_df.to_csv(
    OUTPUT_DIR
    / "model_comparison_by_sku.csv",
    index=False
)


# ============================================================
# 7. 总体模型比较
# ============================================================

overall_model_results = (
    model_results_df
    .groupby("model")
    .agg(
        mape=("mape", "mean"),
        rmse=("rmse", "mean")
    )
    .reset_index()
)

overall_model_results.to_csv(
    OUTPUT_DIR
    / "model_comparison.csv",
    index=False
)

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

for _, row in (
    overall_model_results.iterrows()
):

    print(
        f"{row['model']:<25}"
        f" MAPE: {row['mape']:.2f}%"
        f"   RMSE: {row['rmse']:.2f}"
    )

print()

for sku in sorted(
    sku_model_predictions
):

    print(
        f"{sku}: "
        f"{sku_model_predictions[sku]['best_model']}"
    )


# ============================================================
# 8. 未来30天预测
# ============================================================

forecast_rows = []

inventory_rows = []

latest_date = df["date"].max()


for sku in sorted(
    df["sku"].unique()
):

    sku_df = df[
        df["sku"] == sku
    ].sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    train = sku_df.copy()

    best_model = (
        sku_model_predictions[sku]
        ["best_model"]
    )

    future_dates = pd.date_range(
        start=(
            latest_date
            + pd.Timedelta(days=1)
        ),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    # --------------------------------------------------------
    # Seasonal models
    # --------------------------------------------------------

    (
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
    ) = seasonal_fit(
        train,
        use_promo=(
            best_model
            == "Seasonal + Promo"
        )
    )

    # 假设未来没有已知促销活动
    future_promo = np.zeros(
        FORECAST_HORIZON
    )

    seasonal_forecast = seasonal_predict(
        future_dates,
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
        future_promo
    )

    # --------------------------------------------------------
    # Linear Regression
    # --------------------------------------------------------

    regression_coef = regression_fit(
        train
    )

    regression_forecast = regression_predict(
        future_dates,
        regression_coef,
        future_promo,
        start_t=len(train)
    )

    if best_model == "Linear Regression":

        future_forecast = (
            regression_forecast
        )

    else:

        future_forecast = (
            seasonal_forecast
        )

    # --------------------------------------------------------
    # 保存未来预测
    # --------------------------------------------------------

    for date, forecast in zip(
        future_dates,
        future_forecast
    ):

        forecast_rows.append(
            {
                "sku": sku,
                "date": date,
                "forecast_units": forecast,
                "model": best_model,
            }
        )

    # ========================================================
    # 9. 库存基础参数
    # ========================================================

    avg_daily_demand = (
        future_forecast.mean()
    )

    recent_demand = (
        train["units_sold"]
        .tail(60)
    )

    demand_std = (
        recent_demand.std()
    )

    # --------------------------------------------------------
    # 新增：Annual Demand
    #
    # 这只是为了计算 EOQ。
    # 不改变原来的库存逻辑。
    # --------------------------------------------------------

    annual_demand = (
        avg_daily_demand
        * 365
    )

    # --------------------------------------------------------
    # 新增：EOQ
    #
    # EOQ = sqrt(2DS / H)
    #
    # D = 年需求量
    # S = 每次订货成本
    # H = 每单位每年的持有成本
    #
    # 注意：
    # 这里 EOQ 只是新增的“经济订货量”指标。
    # 暂时不替换原来的 7 天 Review Period。
    # --------------------------------------------------------

    eoq = math.sqrt(
        (
            2
            * annual_demand
            * ORDERING_COST_PER_ORDER
        )
        / HOLDING_COST_PER_UNIT_PER_YEAR
    )

    # EOQ 对应的一年理论订货次数
    eoq_orders_per_year = (
        annual_demand / eoq
    )

    # EOQ 对应的平均订货间隔
    eoq_order_interval_days = (
        365 / eoq_orders_per_year
    )

    # Lead-time demand
    lead_time_demand = (
        avg_daily_demand
        * LEAD_TIME_DAYS
    )

    # Lead-time demand standard deviation
    lead_time_std = (
        demand_std
        * math.sqrt(LEAD_TIME_DAYS)
    )

    # Review-period demand
    review_period_demand = (
        avg_daily_demand
        * REVIEW_PERIOD_DAYS
    )

    # ========================================================
    # 10. 不同 Service Level 的库存优化
    # ========================================================

    for service_level, z in (
        SERVICE_LEVELS.items()
    ):

        # ----------------------------------------------------
        # Safety Stock
        # ----------------------------------------------------

        safety_stock = (
            z
            * lead_time_std
        )

        # ----------------------------------------------------
        # Reorder Point
        # ----------------------------------------------------

        reorder_point = (
            lead_time_demand
            + safety_stock
        )

        # ----------------------------------------------------
        # Order-up-to Level
        # ----------------------------------------------------

        order_up_to_level = (
            lead_time_demand
            + review_period_demand
            + safety_stock
        )

        # ----------------------------------------------------
        # Average Inventory
        #
        # 这里保持原来的逻辑，不使用 EOQ。
        # ----------------------------------------------------

        cycle_inventory = (
            review_period_demand
            / 2
        )

        average_inventory = (
            cycle_inventory
            + safety_stock
        )

        # ====================================================
        # 11. Holding Cost
        # ====================================================

        holding_cost = (
            average_inventory
            * HOLDING_COST_PER_UNIT_PER_YEAR
        )

        # ====================================================
        # 12. Ordering Cost
        #
        # 保持原来的 7 天订货逻辑
        # ====================================================

        orders_per_year = (
            365
            / REVIEW_PERIOD_DAYS
        )

        ordering_cost = (
            orders_per_year
            * ORDERING_COST_PER_ORDER
        )

        # ====================================================
        # 13. Expected Shortage
        # ====================================================

        phi = (
            math.exp(
                -0.5 * z * z
            )
            / math.sqrt(
                2 * math.pi
            )
        )

        Phi = (
            0.5
            * (
                1
                + math.erf(
                    z / math.sqrt(2)
                )
            )
        )

        expected_shortage_per_cycle = (
            lead_time_std
            * (
                phi
                - z * (1 - Phi)
            )
        )

        expected_shortage_per_cycle = max(
            expected_shortage_per_cycle,
            0
        )

        # 一年大约有多少个补货周期
        cycles_per_year = (
            365
            / REVIEW_PERIOD_DAYS
        )

        expected_annual_shortage = (
            expected_shortage_per_cycle
            * cycles_per_year
        )

        # ====================================================
        # 14. Stockout Cost
        # ====================================================

        stockout_cost = (
            expected_annual_shortage
            * STOCKOUT_COST_PER_UNIT
        )

        # ====================================================
        # 15. Total Cost
        #
        # 保持原来的成本模型
        # ====================================================

        total_inventory_cost = (
            holding_cost
            + ordering_cost
            + stockout_cost
        )

        # ====================================================
        # 保存库存结果
        # ====================================================

        inventory_rows.append(
            {
                "sku": sku,

                "category":
                    sku_df["category"].iloc[0],

                "service_level":
                    service_level,

                "model":
                    best_model,

                "avg_daily_demand":
                    avg_daily_demand,

                "demand_std":
                    demand_std,

                # ----------------------------
                # 新增 EOQ 相关字段
                # ----------------------------

                "annual_demand":
                    annual_demand,

                "eoq":
                    eoq,

                "eoq_orders_per_year":
                    eoq_orders_per_year,

                "eoq_order_interval_days":
                    eoq_order_interval_days,

                # ----------------------------
                # 原有库存字段
                # ----------------------------

                "lead_time_demand":
                    lead_time_demand,

                "lead_time_std":
                    lead_time_std,

                "safety_stock":
                    safety_stock,

                "reorder_point":
                    reorder_point,

                "order_up_to_level":
                    order_up_to_level,

                "average_inventory":
                    average_inventory,

                "holding_cost":
                    holding_cost,

                "orders_per_year":
                    orders_per_year,

                "ordering_cost":
                    ordering_cost,

                "expected_shortage_per_cycle":
                    expected_shortage_per_cycle,

                "expected_annual_shortage":
                    expected_annual_shortage,

                "stockout_cost":
                    stockout_cost,

                "total_inventory_cost":
                    total_inventory_cost,
            }
        )


# ============================================================
# 16. 保存 Forecast
# ============================================================

forecast_df = pd.DataFrame(
    forecast_rows
)

forecast_df.to_csv(
    OUTPUT_DIR
    / "forecast_30d.csv",
    index=False
)


# ============================================================
# 17. 保存 Inventory Optimization
# ============================================================

inventory_df = pd.DataFrame(
    inventory_rows
)

inventory_df.to_csv(
    OUTPUT_DIR
    / "inventory_optimization.csv",
    index=False
)


# ============================================================
# 18. Service Level Analysis
# ============================================================

service_level_analysis = (
    inventory_df
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

        total_expected_annual_shortage=(
            "expected_annual_shortage",
            "sum"
        ),

        total_stockout_cost=(
            "stockout_cost",
            "sum"
        ),

        total_inventory_cost=(
            "total_inventory_cost",
            "sum"
        ),
    )
    .reset_index()
)

service_order = [
    "90%",
    "95%",
    "99%"
]

service_level_analysis[
    "service_level"
] = pd.Categorical(
    service_level_analysis[
        "service_level"
    ],
    categories=service_order,
    ordered=True
)

service_level_analysis = (
    service_level_analysis
    .sort_values("service_level")
)

service_level_analysis.to_csv(
    OUTPUT_DIR
    / "service_level_analysis.csv",
    index=False
)


# ============================================================
# 19. 输出 Service Level 结果
# ============================================================

print()
print("=" * 70)
print("SERVICE LEVEL ANALYSIS")
print("=" * 70)

print(
    service_level_analysis[
        [
            "service_level",
            "total_safety_stock",
            "total_average_inventory",
            "total_holding_cost",
            "total_ordering_cost",
            "total_expected_annual_shortage",
            "total_stockout_cost",
            "total_inventory_cost",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 20. 输出95% Service Level 的 SKU 明细
# ============================================================

print()
print("=" * 70)
print("95% SERVICE LEVEL INVENTORY PLAN")
print("=" * 70)

inventory_95 = inventory_df[
    inventory_df["service_level"] == "95%"
].copy()

print(
    inventory_95[
        [
            "sku",
            "category",
            "model",

            "avg_daily_demand",
            "demand_std",

            # 新增
            "annual_demand",
            "eoq",
            "eoq_orders_per_year",
            "eoq_order_interval_days",

            # 原有
            "lead_time_demand",
            "safety_stock",
            "reorder_point",
            "order_up_to_level",
            "average_inventory",
            "holding_cost",
            "stockout_cost",
            "total_inventory_cost",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 21. EOQ 汇总
# ============================================================

print()
print("=" * 70)
print("EOQ ANALYSIS")
print("=" * 70)

print(
    inventory_95[
        [
            "sku",
            "annual_demand",
            "eoq",
            "eoq_orders_per_year",
            "eoq_order_interval_days",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:,.2f}"
    )
)


# ============================================================
# 22. 画 Service Level vs Total Cost
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    service_level_analysis[
        "service_level"
    ].astype(str),
    service_level_analysis[
        "total_inventory_cost"
    ],
    marker="o"
)

plt.xlabel(
    "Service Level"
)

plt.ylabel(
    "Total Inventory Cost"
)

plt.title(
    "Service Level vs Total Inventory Cost"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "service_level_vs_cost.png",
    dpi=150
)

plt.close()


# ============================================================
# 23. 画 Holding / Stockout / Ordering Cost
# ============================================================

plt.figure(
    figsize=(9, 6)
)

x = np.arange(
    len(service_level_analysis)
)

width = 0.25

plt.bar(
    x - width,
    service_level_analysis[
        "total_holding_cost"
    ],
    width,
    label="Holding Cost"
)

plt.bar(
    x,
    service_level_analysis[
        "total_stockout_cost"
    ],
    width,
    label="Stockout Cost"
)

plt.bar(
    x + width,
    service_level_analysis[
        "total_ordering_cost"
    ],
    width,
    label="Ordering Cost"
)

plt.xticks(
    x,
    service_level_analysis[
        "service_level"
    ].astype(str)
)

plt.xlabel(
    "Service Level"
)

plt.ylabel(
    "Cost"
)

plt.title(
    "Inventory Cost Components"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "inventory_cost_components.png",
    dpi=150
)

plt.close()


# ============================================================
# 24. 画 EOQ
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.bar(
    inventory_95["sku"],
    inventory_95["eoq"]
)

plt.xlabel(
    "SKU"
)

plt.ylabel(
    "EOQ (units)"
)

plt.title(
    "Economic Order Quantity by SKU"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "eoq_by_sku.png",
    dpi=150
)

plt.close()


# ============================================================
# 25. 保存 JSON Summary
# ============================================================

best_overall_model = (
    overall_model_results
    .sort_values("mape")
    .iloc[0]
)

summary = {
    "skus": int(
        df["sku"].nunique()
    ),

    "days": int(
        df["date"].nunique()
    ),

    "backtest_days":
        BACKTEST_DAYS,

    "forecast_horizon_days":
        FORECAST_HORIZON,

    "lead_time_days":
        LEAD_TIME_DAYS,

    "review_period_days":
        REVIEW_PERIOD_DAYS,

    "best_overall_model":
        best_overall_model["model"],

    "best_overall_mape":
        round(
            float(
                best_overall_model["mape"]
            ),
            2
        ),

    "holding_cost_per_unit_per_year":
        HOLDING_COST_PER_UNIT_PER_YEAR,

    "stockout_cost_per_unit":
        STOCKOUT_COST_PER_UNIT,

    "ordering_cost_per_order":
        ORDERING_COST_PER_ORDER,

    # 新增
    "eoq_added":
        True,
}


with open(
    OUTPUT_DIR / "results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 26. 完成
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Results saved to: {OUTPUT_DIR}"
)

print()
print("Generated files:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f" - {file.name}"
    )