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


# ============================================================
# 2. Service Level
# ============================================================

SERVICE_LEVELS = {
    "90%": 1.28,
    "95%": 1.65,
    "99%": 2.33,
}


# ============================================================
# 3. 成本参数
# ============================================================
#
# 这些是项目中的示例成本假设。
#
# Holding Cost:
# 每单位库存每年需要承担的持有成本
#
# Stockout Cost:
# 每缺货一个单位造成的成本
#
# Ordering Cost:
# 每下一次订单产生的固定成本
# ============================================================

HOLDING_COST_PER_UNIT_PER_YEAR = 5.00
STOCKOUT_COST_PER_UNIT = 5.00
ORDERING_COST_PER_ORDER = 100.00


# ============================================================
# 4. 读取数据
# ============================================================

df = pd.read_csv(
    DATA_FILE,
    parse_dates=["date"]
)

df = (
    df
    .sort_values(["sku", "date"])
    .reset_index(drop=True)
)


# ============================================================
# 5. 评价指标
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
# 6. Seasonal Model
# ============================================================
#
# 这个版本恢复之前使用的逻辑：
#
# demand
# =
# trend
# + weekly seasonality
# + monthly seasonality
# + promotion effect
#
# 注意：
# trend 使用训练数据真实日期对应的位置。
# 未来预测时，继续沿用训练数据最后一个时间点之后的位置。
# ============================================================

def seasonal_fit(train_df, use_promo=False):

    data = train_df.copy()

    data["dow"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    x = np.arange(len(data))

    y = data["units_sold"].values

    if len(data) >= 2:

        trend_coef = np.polyfit(
            x,
            y,
            1
        )

    else:

        trend_coef = [
            0,
            y.mean()
        ]

    trend = np.polyval(
        trend_coef,
        x
    )

    # --------------------------------------------------------
    # 去掉趋势之后，分析季节性
    # --------------------------------------------------------

    residual = (
        data["units_sold"].values
        - trend
    )

    residual_series = pd.Series(
        residual,
        index=data.index
    )

    # --------------------------------------------------------
    # Weekly seasonality
    # --------------------------------------------------------

    dow_factor = (
        residual_series
        .groupby(data["dow"])
        .mean()
    )

    # --------------------------------------------------------
    # Monthly seasonality
    # --------------------------------------------------------

    month_factor = (
        residual_series
        .groupby(data["month"])
        .mean()
    )

    # --------------------------------------------------------
    # Promotion effect
    # --------------------------------------------------------

    promo_effect = 0.0

    if use_promo and "promo" in data.columns:

        promo_mask = (
            data["promo"] == 1
        )

        non_promo_mask = (
            data["promo"] == 0
        )

        if (
            promo_mask.sum() > 0
            and non_promo_mask.sum() > 0
        ):

            promo_effect = (
                data.loc[
                    promo_mask,
                    "units_sold"
                ].mean()
                -
                data.loc[
                    non_promo_mask,
                    "units_sold"
                ].mean()
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
    promo=None,
    start_t=0
):

    dates = pd.Series(
        pd.to_datetime(dates)
    )

    # --------------------------------------------------------
    # 关键：
    #
    # start_t 表示这些日期在整个时间序列中的位置。
    #
    # 回测：
    # start_t = len(train)
    #
    # 未来30天：
    # start_t = len(train)
    # --------------------------------------------------------

    x_future = (
        np.arange(len(dates))
        + start_t
    )

    trend = np.polyval(
        trend_coef,
        x_future
    )

    dow = dates.dt.dayofweek

    month = dates.dt.month

    dow_component = (
        dow
        .map(dow_factor)
        .fillna(0)
        .values
    )

    month_component = (
        month
        .map(month_factor)
        .fillna(0)
        .values
    )

    prediction = (
        trend
        + dow_component
        + month_component
    )

    if promo is not None:

        promo = np.asarray(promo)

        prediction = (
            prediction
            + promo * promo_effect
        )

    return np.maximum(
        prediction,
        0
    )


# ============================================================
# 7. Linear Regression
# ============================================================

def regression_fit(train_df):

    data = train_df.copy()

    data["t"] = np.arange(
        len(data)
    )

    data["dow"] = (
        data["date"]
        .dt.dayofweek
    )

    data["month"] = (
        data["date"]
        .dt.month
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
        dates
        .dt.dayofweek
        .values
    )

    month = (
        dates
        .dt.month
        .values
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
# 8. Model Backtest
# ============================================================

model_results = []

sku_model_predictions = {}


for sku in sorted(
    df["sku"].unique()
):

    sku_df = (
        df[df["sku"] == sku]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    train = sku_df.iloc[
        :-BACKTEST_DAYS
    ].copy()

    test = sku_df.iloc[
        -BACKTEST_DAYS:
    ].copy()

    # ========================================================
    # Model 1
    # Seasonal Baseline
    # ========================================================

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
        month_factor,
        promo_effect=0,
        promo=None,
        start_t=len(train)
    )

    seasonal_mape = mape(
        test["units_sold"],
        pred_seasonal
    )

    seasonal_rmse = rmse(
        test["units_sold"],
        pred_seasonal
    )

    # ========================================================
    # Model 2
    # Seasonal + Promo
    # ========================================================

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
        test["promo"].values,
        start_t=len(train)
    )

    promo_mape = mape(
        test["units_sold"],
        pred_promo
    )

    promo_rmse = rmse(
        test["units_sold"],
        pred_promo
    )

    # ========================================================
    # Model 3
    # Linear Regression
    # ========================================================

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

    # ========================================================
    # 汇总
    # ========================================================

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

    # --------------------------------------------------------
    # 选择当前 SKU 的最佳模型
    # --------------------------------------------------------

    best_model = min(
        models,
        key=lambda name:
            models[name][0]
    )

    sku_model_predictions[sku] = {
        "best_model": best_model,
        "models": models,
    }

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    for model_name, (
        model_mape,
        model_rmse,
        predictions
    ) in models.items():

        model_results.append(
            {
                "sku": sku,
                "model": model_name,
                "mape": model_mape,
                "rmse": model_rmse,
                "is_best":
                    model_name == best_model,
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
# 9. Overall Model Comparison
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

# 按 MAPE 从低到高排列
overall_model_results = (
    overall_model_results
    .sort_values("mape")
)

for _, row in (
    overall_model_results
    .iterrows()
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
# 10. Future Forecast
# ============================================================

forecast_rows = []

inventory_rows = []

latest_date = df["date"].max()


for sku in sorted(
    df["sku"].unique()
):

    sku_df = (
        df[df["sku"] == sku]
        .sort_values("date")
        .reset_index(drop=True)
    )

    train = sku_df.copy()

    best_model = (
        sku_model_predictions[sku]
        ["best_model"]
    )

    future_dates = pd.date_range(
        start=latest_date
        + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D"
    )

    # ========================================================
    # Seasonal Models
    # ========================================================

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

    # --------------------------------------------------------
    # 未来促销信息未知
    # 暂时假设未来30天没有促销
    # --------------------------------------------------------

    future_promo = np.zeros(
        FORECAST_HORIZON
    )

    seasonal_forecast = seasonal_predict(
        future_dates,
        trend_coef,
        dow_factor,
        month_factor,
        promo_effect,
        future_promo,
        start_t=len(train)
    )

    # ========================================================
    # Linear Regression
    # ========================================================

    regression_coef = regression_fit(
        train
    )

    regression_forecast = regression_predict(
        future_dates,
        regression_coef,
        future_promo,
        start_t=len(train)
    )

    # ========================================================
    # 根据 SKU 最佳模型选择最终预测
    # ========================================================

    if best_model == "Linear Regression":

        future_forecast = (
            regression_forecast
        )

    else:

        future_forecast = (
            seasonal_forecast
        )

    # ========================================================
    # 保存30天预测
    # ========================================================

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
    # 11. Inventory Parameters
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
    # Lead Time Demand
    # --------------------------------------------------------

    lead_time_demand = (
        avg_daily_demand
        * LEAD_TIME_DAYS
    )

    # --------------------------------------------------------
    # Lead Time Demand Std
    # --------------------------------------------------------

    lead_time_std = (
        demand_std
        * math.sqrt(
            LEAD_TIME_DAYS
        )
    )

    # --------------------------------------------------------
    # Review Period Demand
    # --------------------------------------------------------

    review_period_demand = (
        avg_daily_demand
        * REVIEW_PERIOD_DAYS
    )

    # ========================================================
    # 12. Service Level Analysis
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
        # ----------------------------------------------------

        cycle_inventory = (
            review_period_demand / 2
        )

        average_inventory = (
            cycle_inventory
            + safety_stock
        )

        # ====================================================
        # 13. Holding Cost
        # ====================================================

        holding_cost = (
            average_inventory
            * HOLDING_COST_PER_UNIT_PER_YEAR
        )

        # ====================================================
        # 14. Ordering Cost
        # ====================================================
        #
        # 每7天补一次货
        #
        # 一年订单次数：
        #
        # 365 / 7
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
        # 15. Expected Shortage
        # ====================================================
        #
        # Lead-Time Demand
        # 假设近似服从 Normal Distribution
        #
        # Expected Shortage:
        #
        # sigma × [phi(z)
        #       - z × (1 - Phi(z))]
        #
        # phi = Standard Normal PDF
        # Phi = Standard Normal CDF
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
                    z
                    / math.sqrt(2)
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

        # ----------------------------------------------------
        # 每年补货周期
        # ----------------------------------------------------

        cycles_per_year = (
            365
            / REVIEW_PERIOD_DAYS
        )

        expected_annual_shortage = (
            expected_shortage_per_cycle
            * cycles_per_year
        )

        # ====================================================
        # 16. Stockout Cost
        # ====================================================

        stockout_cost = (
            expected_annual_shortage
            * STOCKOUT_COST_PER_UNIT
        )

        # ====================================================
        # 17. Total Inventory Cost
        # ====================================================

        total_inventory_cost = (
            holding_cost
            + stockout_cost
            + ordering_cost
        )

        inventory_rows.append(
            {
                "sku": sku,
                "category":
                    sku_df[
                        "category"
                    ].iloc[0],

                "model":
                    best_model,

                "service_level":
                    service_level,

                "avg_daily_demand":
                    avg_daily_demand,

                "demand_std":
                    demand_std,

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
# 18. 保存 Forecast
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
# 19. 保存 Inventory Optimization
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
# 20. Service Level Summary
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
# 21. 输出 Service Level Analysis
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
        float_format=lambda x:
            f"{x:,.2f}"
    )
)


# ============================================================
# 22. 95% Service Level Inventory Plan
# ============================================================

print()
print("=" * 70)
print("95% SERVICE LEVEL INVENTORY PLAN")
print("=" * 70)


inventory_95 = inventory_df[
    inventory_df["service_level"]
    == "95%"
].copy()


print(
    inventory_95[
        [
            "sku",
            "category",
            "model",
            "avg_daily_demand",
            "demand_std",
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
        float_format=lambda x:
            f"{x:,.2f}"
    )
)


# ============================================================
# 23. Service Level vs Total Cost Chart
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
# 24. Cost Components Chart
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
# 25. Summary JSON
# ============================================================

best_overall_model = (
    overall_model_results
    .sort_values("mape")
    .iloc[0]
)


summary = {

    "skus":
        int(
            df["sku"].nunique()
        ),

    "days":
        int(
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
        best_overall_model[
            "model"
        ],

    "best_overall_mape":
        round(
            float(
                best_overall_model[
                    "mape"
                ]
            ),
            2
        ),

    "holding_cost_per_unit_per_year":
        HOLDING_COST_PER_UNIT_PER_YEAR,

    "stockout_cost_per_unit":
        STOCKOUT_COST_PER_UNIT,

    "ordering_cost_per_order":
        ORDERING_COST_PER_ORDER,
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