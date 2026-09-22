
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# 1. Page Configuration
# 页面配置
# ============================================================

st.set_page_config(
    page_title="AI Supply Chain | AI供应链",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# 2. File Paths
# 文件路径
# ============================================================

BASE = Path(__file__).resolve().parent

FORECAST_FILE = BASE / "forecast_30d.csv"
INVENTORY_FILE = BASE / "inventory_recommendations.csv"
MODEL_FILE = BASE / "model_comparison.csv"
SERVICE_FILE = BASE / "service_level_analysis.csv"
RESULTS_FILE = BASE / "results.json"


# ============================================================
# 3. Load Data
# 加载数据
# ============================================================

@st.cache_data
def load_data():

    forecast = pd.read_csv(FORECAST_FILE)

    inventory = pd.read_csv(INVENTORY_FILE)

    model_comparison = pd.read_csv(MODEL_FILE)

    service_level = pd.read_csv(SERVICE_FILE)

    results = {}

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)

    return (
        forecast,
        inventory,
        model_comparison,
        service_level,
        results,
    )


forecast_df, inventory_df, model_df, service_df, results = load_data()


# ============================================================
# 4. Data Preparation
# 数据预处理
# ============================================================

forecast_df["date"] = pd.to_datetime(
    forecast_df["date"]
)

skus = sorted(
    forecast_df["sku"].unique()
)


# ============================================================
# 5. Convert Model Comparison Data
# 将模型比较CSV从宽表转换成长表
# ============================================================

model_long_df = pd.DataFrame()

if not model_df.empty:

    required_model_columns = [
        "sku",
        "seasonal_mape",
        "promo_mape",
        "regression_mape",
    ]

    if all(
        col in model_df.columns
        for col in required_model_columns
    ):

        model_long_df = model_df.melt(
            id_vars=["sku"],
            value_vars=[
                "seasonal_mape",
                "promo_mape",
                "regression_mape",
            ],
            var_name="model",
            value_name="mape",
        )

        model_name_mapping = {

            "seasonal_mape":
                "Seasonal Baseline",

            "promo_mape":
                "Seasonal + Promo",

            "regression_mape":
                "Linear Regression",
        }

        model_long_df["model"] = (
            model_long_df["model"]
            .map(model_name_mapping)
        )


# ============================================================
# 6. Determine Best Model
# 确定最佳模型
# ============================================================

best_model = "Seasonal + Promo"
best_mape = 8.21

if not model_long_df.empty:

    grouped_models = (
        model_long_df
        .groupby("model")["mape"]
        .mean()
        .sort_values()
    )

    if not grouped_models.empty:

        best_model = grouped_models.index[0]

        best_mape = grouped_models.iloc[0]


# ============================================================
# 7. Sidebar
# 侧边栏
# ============================================================

st.sidebar.title(
    "📦 Supply Chain Control Center"
)

st.sidebar.caption(
    "供应链控制中心"
)

st.sidebar.markdown(
    "### Navigation（导航）"
)

page = st.sidebar.radio(
    "Select Page（选择页面）",
    [
        "Overview（总览）",
        "Demand Forecast（需求预测）",
        "Inventory Optimization（库存优化）",
        "Model Comparison（模型比较）",
    ],
)

st.sidebar.markdown("---")

selected_sku = st.sidebar.selectbox(
    "Select SKU（选择产品）",
    skus,
)


# ============================================================
# 8. Overview
# 总览页面
# ============================================================

if page == "Overview（总览）":

    st.title(
        "📦 AI-Driven Supply Chain Demand Forecasting & Inventory Optimization"
    )

    st.caption(
        "AI驱动的供应链需求预测与库存优化系统"
    )

    st.markdown(
        """
        This dashboard integrates demand forecasting,
        model comparison, and inventory optimization
        into one supply chain decision-support system.

        本系统将需求预测、模型比较和库存优化整合到
        一个供应链决策支持平台中。
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI Cards
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "SKUs",
            len(skus),
            help="Number of products / 产品数量",
        )

    with col2:

        st.metric(
            "Best Model",
            str(best_model),
            help="Lowest average MAPE / 平均MAPE最低",
        )

    with col3:

        st.metric(
            "Best MAPE",
            f"{best_mape:.2f}%",
            help="Mean Absolute Percentage Error / 平均绝对百分比误差",
        )

    with col4:

        st.metric(
            "Forecast Horizon",
            "30 Days",
            help="Future forecast period / 未来预测周期",
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Overall Demand Forecast
    # 总体需求预测
    # --------------------------------------------------------

    st.subheader(
        "📈 Overall Demand Forecast（总体需求预测）"
    )

    overall_forecast = (
        forecast_df
        .groupby(
            "date",
            as_index=False
        )["forecast_units"]
        .sum()
    )

    fig = px.line(
        overall_forecast,
        x="date",
        y="forecast_units",
        title="Total Forecasted Demand（总体预测需求）",
        labels={
            "date": "Date（日期）",
            "forecast_units":
                "Forecast Units（预测需求量）",
        },
    )

    fig.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Model Summary
    # --------------------------------------------------------

    st.subheader(
        "🤖 Model Summary（模型摘要）"
    )

    if not model_long_df.empty:

        model_summary = (
            model_long_df
            .groupby("model")["mape"]
            .mean()
            .reset_index()
            .sort_values("mape")
        )

        model_summary = model_summary.rename(
            columns={
                "model": "Model（模型）",
                "mape": "Average MAPE（平均MAPE）",
            }
        )

        st.dataframe(
            model_summary,
            use_container_width=True,
        )

    else:

        st.info(
            "No model comparison data available. "
            "暂无模型比较数据。"
        )


# ============================================================
# 9. Demand Forecast
# 需求预测页面
# ============================================================

elif page == "Demand Forecast（需求预测）":

    st.title(
        "📈 Demand Forecast（需求预测）"
    )

    st.caption(
        f"SKU: {selected_sku} | "
        "30-Day Demand Forecast（30天需求预测）"
    )

    sku_forecast = forecast_df[
        forecast_df["sku"] == selected_sku
    ].copy()

    sku_forecast = sku_forecast.sort_values(
        "date"
    )

    # --------------------------------------------------------
    # Forecast Chart
    # --------------------------------------------------------

    fig = px.line(
        sku_forecast,
        x="date",
        y="forecast_units",
        color="model",
        markers=True,
        title=(
            f"{selected_sku} Demand Forecast"
            "（需求预测）"
        ),
        labels={
            "date": "Date（日期）",
            "forecast_units":
                "Forecast Units（预测需求量）",
            "model": "Model（模型）",
        },
    )

    fig.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Forecast Statistics
    # --------------------------------------------------------

    st.subheader(
        "📊 Forecast Statistics（预测统计）"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        avg_forecast = (
            sku_forecast["forecast_units"]
            .mean()
        )

        st.metric(
            "Average Forecast",
            f"{avg_forecast:.1f}",
            help="Average daily forecast / 平均每日预测需求",
        )

    with col2:

        max_forecast = (
            sku_forecast["forecast_units"]
            .max()
        )

        st.metric(
            "Peak Forecast",
            f"{max_forecast:.1f}",
            help="Maximum forecast / 最大预测需求",
        )

    with col3:

        min_forecast = (
            sku_forecast["forecast_units"]
            .min()
        )

        st.metric(
            "Minimum Forecast",
            f"{min_forecast:.1f}",
            help="Minimum forecast / 最低预测需求",
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Forecast Table
    # --------------------------------------------------------

    st.subheader(
        "📋 Forecast Data（预测数据）"
    )

    display_forecast = sku_forecast.copy()

    display_forecast["date"] = (
        display_forecast["date"]
        .dt.strftime("%Y-%m-%d")
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
    )


# ============================================================
# 10. Inventory Optimization
# 库存优化页面
# ============================================================

elif page == "Inventory Optimization（库存优化）":

    st.title(
        "📦 Inventory Optimization（库存优化）"
    )

    st.caption(
        f"SKU: {selected_sku} | "
        "Inventory Planning（库存计划）"
    )

    # --------------------------------------------------------
    # Service Level Selector
    # --------------------------------------------------------

    service_level_options = [
        0.90,
        0.95,
        0.99,
    ]

    selected_service = st.selectbox(
        "Select Service Level（选择服务水平）",
        service_level_options,
        index=1,
        format_func=lambda x: f"{x:.0%}",
    )

    # --------------------------------------------------------
    # Service Level Summary
    # --------------------------------------------------------

    selected_service_row = service_df[
        service_df["service_level"].round(2)
        == round(selected_service, 2)
    ]

    if not selected_service_row.empty:

        service_row = selected_service_row.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Safety Stock",
                f"{service_row['total_safety_stock']:.0f}",
            )

        with col2:

            st.metric(
                "Total Reorder Point",
                f"{service_row['total_reorder_point']:.0f}",
            )

        with col3:

            st.metric(
                "Average Inventory",
                f"{service_row['total_average_inventory']:.0f}",
            )

        with col4:

            st.metric(
                "Total Inventory Cost",
                f"{service_row['total_inventory_cost']:.0f}",
            )

    st.markdown("---")

    # --------------------------------------------------------
    # SKU Inventory Recommendation
    # --------------------------------------------------------

    sku_inventory = inventory_df[
        (
            inventory_df["sku"]
            == selected_sku
        )
        &
        (
            inventory_df["service_level"].round(2)
            == round(selected_service, 2)
        )
    ].copy()

    if not sku_inventory.empty:

        row = sku_inventory.iloc[0]

        st.subheader(
            f"🎯 {selected_sku} Inventory Recommendation"
        )

        st.caption(
            "库存决策建议"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Safety Stock（安全库存）",
                f"{row['safety_stock']:.0f}",
            )

        with col2:

            st.metric(
                "Reorder Point（再订货点）",
                f"{row['reorder_point']:.0f}",
            )

        with col3:

            st.metric(
                "Order-Up-To Level（补货上限）",
                f"{row['order_up_to_level']:.0f}",
            )

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "EOQ（经济订货量）",
                f"{row['eoq']:.0f}",
            )

        with col2:

            st.metric(
                "EOQ Interval",
                f"{row['eoq_order_interval_days']:.1f} days",
                help=(
                    "Theoretical economic ordering interval / "
                    "理论经济订货间隔"
                ),
            )

        with col3:

            st.metric(
                "Daily Demand",
                f"{row['avg_daily_demand']:.1f}",
            )

        st.markdown("---")

        # ----------------------------------------------------
        # Detailed Inventory Table
        # ----------------------------------------------------

        st.subheader(
            "📋 Inventory Planning Details"
            "（库存计划明细）"
        )

        columns_to_show = [

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

            "model",
        ]

        available_columns = [
            col
            for col in columns_to_show
            if col in sku_inventory.columns
        ]

        st.dataframe(
            sku_inventory[available_columns],
            use_container_width=True,
        )

    else:

        st.warning(
            "No inventory recommendation found for this "
            "SKU and service level."
            "该SKU和服务水平没有找到库存建议。"
        )


# ============================================================
# 11. Model Comparison
# 模型比较页面
# ============================================================

elif page == "Model Comparison（模型比较）":

    st.title(
        "🤖 Model Comparison（模型比较）"
    )

    st.caption(
        "Compare forecasting models using MAPE"
        "（使用MAPE比较预测模型）"
    )

    if not model_long_df.empty:

        # ----------------------------------------------------
        # Overall Model Performance
        # ----------------------------------------------------

        st.subheader(
            "📊 Overall Model Performance"
            "（总体模型表现）"
        )

        model_summary = (
            model_long_df
            .groupby("model")
            .agg(
                MAPE=("mape", "mean")
            )
            .reset_index()
            .sort_values("MAPE")
        )

        st.dataframe(
            model_summary,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # MAPE Chart
        # ----------------------------------------------------

        fig_mape = px.bar(
            model_summary,
            x="model",
            y="MAPE",
            title=(
                "Average MAPE by Model"
                "（各模型平均MAPE）"
            ),
            labels={
                "model": "Model（模型）",
                "MAPE": "MAPE (%)",
            },
        )

        st.plotly_chart(
            fig_mape,
            use_container_width=True,
        )

        st.markdown("---")

        # ----------------------------------------------------
        # SKU-Level Model Comparison
        # ----------------------------------------------------

        st.subheader(
            f"🔎 {selected_sku} Model Comparison"
        )

        st.caption(
            "SKU级别模型比较"
        )

        sku_model_df = model_long_df[
            model_long_df["sku"]
            == selected_sku
        ].copy()

        if not sku_model_df.empty:

            st.dataframe(
                sku_model_df,
                use_container_width=True,
            )

            fig_sku = px.bar(
                sku_model_df,
                x="model",
                y="mape",
                title=(
                    f"{selected_sku} MAPE Comparison"
                    "（MAPE比较）"
                ),
                labels={
                    "model": "Model（模型）",
                    "mape": "MAPE (%)",
                },
            )

            st.plotly_chart(
                fig_sku,
                use_container_width=True,
            )

            # ------------------------------------------------
            # Best Model for Selected SKU
            # ------------------------------------------------

            best_sku_row = (
                sku_model_df
                .sort_values("mape")
                .iloc[0]
            )

            st.success(
                f"Best model for {selected_sku}: "
                f"{best_sku_row['model']} | "
                f"MAPE: {best_sku_row['mape']:.2f}%"
            )

        else:

            st.info(
                "No model comparison data for this SKU."
                "该SKU没有模型比较数据。"
            )

    else:

        st.warning(
            "Model comparison data is unavailable."
            "模型比较数据不可用。"
        )


# ============================================================
# 12. Sidebar Footer
# 侧边栏页脚
# ============================================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI-Driven Supply Chain Demand Forecasting & Inventory Optimization"
)

st.sidebar.caption(
    "AI驱动的供应链需求预测与库存优化系统"
)

