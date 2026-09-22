
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from openai import OpenAI

if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Supply Chain Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

# ============================================================
# QWEN LLM CLIENT
# ============================================================

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

MODEL_NAME = "qwen3.8-max"
# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    forecast_df = pd.read_csv(
        BASE / "forecast_30d.csv",
        parse_dates=["date"]
    )

    inventory_df = pd.read_csv(
        BASE / "inventory_recommendations.csv"
    )

    model_df = pd.read_csv(
        BASE / "model_comparison.csv"
    )

    service_df = pd.read_csv(
        BASE / "service_level_analysis.csv"
    )

    return forecast_df, inventory_df, model_df, service_df


forecast_df, inventory_df, model_df, service_df = load_data()


# ============================================================
# MODEL COMPARISON DATA
# ============================================================

model_long_df = model_df.melt(
    id_vars=["sku"],
    value_vars=[
        "seasonal_mape",
        "promo_mape",
        "regression_mape"
    ],
    var_name="model",
    value_name="mape"
)

model_name_mapping = {
    "seasonal_mape": "Seasonal Baseline",
    "promo_mape": "Seasonal + Promo",
    "regression_mape": "Linear Regression"
}

model_long_df["model"] = model_long_df["model"].map(
    model_name_mapping
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📦 Supply Chain System")

page = st.sidebar.radio(
    "Navigation / 页面导航",
    [
        "Overview / 总览",
        "Demand Forecast / 需求预测",
        "Inventory Optimization / 库存优化",
        "Model Comparison / 模型比较",
        "AI Supply Chain Copilot / AI供应链助手"
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview / 总览":

    st.title(
        "AI-Driven Supply Chain Demand Forecasting & Inventory Optimization"
    )

    st.subheader(
        "AI驱动的供应链需求预测与库存优化系统"
    )

    st.markdown(
        """
        This system combines demand forecasting, inventory optimization,
        and model comparison to support supply chain planning decisions.

        本系统结合需求预测、库存优化和模型比较，
        为供应链计划决策提供支持。
        """
    )

    sku_count = forecast_df["sku"].nunique()

    forecast_days = forecast_df["date"].nunique()

    best_model_name = (
        model_long_df.groupby("model")["mape"]
        .mean()
        .sort_values()
        .index[0]
    )

    best_model_mape = (
        model_long_df.groupby("model")["mape"]
        .mean()
        .sort_values()
        .iloc[0]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "SKUs",
            sku_count
        )

    with col2:
        st.metric(
            "Forecast Horizon",
            f"{forecast_days} days"
        )

    with col3:
        st.metric(
            "Best Model",
            best_model_name
        )

    with col4:
        st.metric(
            "Average MAPE",
            f"{best_model_mape:.2f}%"
        )

    st.divider()

    st.subheader(
        "Model Performance / 模型表现"
    )

    model_summary = (
        model_long_df
        .groupby("model")["mape"]
        .mean()
        .reset_index()
        .sort_values("mape")
    )

    st.dataframe(
        model_summary,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "MAPE = Mean Absolute Percentage Error，"
        "用于衡量预测值与实际值之间的平均百分比误差。"
    )


# ============================================================
# DEMAND FORECAST
# ============================================================

elif page == "Demand Forecast / 需求预测":

    st.title(
        "Demand Forecast / 需求预测"
    )

    selected_sku = st.selectbox(
        "Select SKU / 选择 SKU",
        sorted(forecast_df["sku"].unique())
    )

    sku_forecast = forecast_df[
        forecast_df["sku"] == selected_sku
    ].copy()

    st.subheader(
        f"{selected_sku} Forecast / {selected_sku} 需求预测"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    for model_name in sku_forecast["model"].unique():

        model_data = sku_forecast[
            sku_forecast["model"] == model_name
        ]

        ax.plot(
            model_data["date"],
            model_data["forecast_units"],
            label=model_name
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Units")

    ax.set_title(
        f"{selected_sku} - 30 Day Demand Forecast"
    )

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    st.subheader(
        "Model Performance / 模型表现"
    )

    sku_model_performance = model_long_df[
        model_long_df["sku"] == selected_sku
    ].copy()

    sku_model_performance = sku_model_performance.sort_values(
        "mape"
    )

    st.dataframe(
        sku_model_performance,
        use_container_width=True,
        hide_index=True
    )

    best_sku_model = sku_model_performance.iloc[0]["model"]

    best_sku_mape = sku_model_performance.iloc[0]["mape"]

    st.success(
        f"Best model for {selected_sku}: "
        f"{best_sku_model} "
        f"(MAPE: {best_sku_mape:.2f}%)"
    )


# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

elif page == "Inventory Optimization / 库存优化":

    st.title(
        "Inventory Optimization / 库存优化"
    )

    selected_sku = st.selectbox(
        "Select SKU / 选择 SKU",
        sorted(inventory_df["sku"].unique())
    )

    available_service_levels = sorted(
        inventory_df["service_level"]
        .dropna()
        .unique()
    )

    selected_service = st.selectbox(
        "Service Level / 服务水平",
        available_service_levels,
        index=(
            available_service_levels.index("95%")
            if "95%" in available_service_levels
            else 0
        )
    )

    sku_inventory = inventory_df[
        (inventory_df["sku"] == selected_sku) &
        (inventory_df["service_level"] == selected_service)
    ].copy()

    if sku_inventory.empty:

        st.warning(
            "No inventory recommendation found for "
            "this SKU and service level."
        )

    else:

        row = sku_inventory.iloc[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Safety Stock",
                f"{row['safety_stock']:.1f}"
            )

        with col2:
            st.metric(
                "Reorder Point",
                f"{row['reorder_point']:.1f}"
            )

        with col3:
            st.metric(
                "Order-Up-To Level",
                f"{row['order_up_to_level']:.1f}"
            )

        with col4:
            st.metric(
                "EOQ",
                f"{row['eoq']:.1f}"
            )

        st.divider()

        st.subheader(
            "Inventory Policy / 库存策略"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Average Daily Demand:** "
                f"{row['avg_daily_demand']:.2f}"
            )

            st.write(
                f"**Demand Standard Deviation:** "
                f"{row['demand_std']:.2f}"
            )

            st.write(
                f"**Annual Demand:** "
                f"{row['annual_demand']:.0f}"
            )

            st.write(
                f"**Lead-Time Demand:** "
                f"{row['lead_time_demand']:.2f}"
            )

            st.write(
                f"**Safety Stock:** "
                f"{row['safety_stock']:.2f}"
            )

        with col2:

            st.write(
                f"**Reorder Point:** "
                f"{row['reorder_point']:.2f}"
            )

            st.write(
                f"**Order-Up-To Level:** "
                f"{row['order_up_to_level']:.2f}"
            )

            st.write(
                f"**Average Inventory:** "
                f"{row['average_inventory']:.2f}"
            )

            st.write(
                f"**EOQ:** "
                f"{row['eoq']:.2f}"
            )

            st.write(
                f"**EOQ Order Interval:** "
                f"{row['eoq_order_interval_days']:.2f} days"
            )

        st.divider()

        st.subheader(
            "Inventory Cost / 库存成本"
        )

        cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)

        with cost_col1:
            st.metric(
                "Holding Cost",
                f"${row['holding_cost']:,.2f}"
            )

        with cost_col2:
            st.metric(
                "Ordering Cost",
                f"${row['ordering_cost']:,.2f}"
            )

        with cost_col3:
            st.metric(
                "Stockout Cost",
                f"${row['stockout_cost']:,.2f}"
            )

        with cost_col4:
            st.metric(
                "Total Inventory Cost",
                f"${row['total_inventory_cost']:,.2f}"
            )

        st.divider()

        st.subheader(
            "Model Used / 使用模型"
        )

        st.success(
            f"{row['model']}"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison / 模型比较":

    st.title(
        "Model Comparison / 模型比较"
    )

    st.markdown(
        """
        The system compares three forecasting approaches:

        - Seasonal Baseline
        - Seasonal + Promotion
        - Linear Regression

        系统比较三种需求预测方法。
        """
    )

    overall_model_performance = (
        model_long_df
        .groupby("model")["mape"]
        .mean()
        .reset_index()
        .sort_values("mape")
    )

    st.subheader(
        "Average MAPE by Model / 各模型平均 MAPE"
    )

    st.dataframe(
        overall_model_performance,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        overall_model_performance["model"],
        overall_model_performance["mape"]
    )

    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Model")

    ax.set_title(
        "Average MAPE by Forecasting Model"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    st.pyplot(fig)

    st.subheader(
        "SKU-Level Model Comparison / SKU级模型比较"
    )

    selected_sku = st.selectbox(
        "Select SKU / 选择 SKU",
        sorted(model_long_df["sku"].unique()),
        key="model_comparison_sku"
    )

    sku_comparison = model_long_df[
        model_long_df["sku"] == selected_sku
    ].copy()

    sku_comparison = sku_comparison.sort_values(
        "mape"
    )

    st.dataframe(
        sku_comparison,
        use_container_width=True,
        hide_index=True
    )

    best_model = sku_comparison.iloc[0]["model"]

    best_mape = sku_comparison.iloc[0]["mape"]

    st.success(
        f"Best model for {selected_sku}: "
        f"{best_model} "
        f"(MAPE: {best_mape:.2f}%)"
    )


# ============================================================
# AI SUPPLY CHAIN COPILOT
# ============================================================

elif page == "AI Supply Chain Copilot / AI供应链助手":

    st.title(
        "🤖 AI Supply Chain Copilot"
    )

    st.subheader(
        "AI供应链助手"
    )

    st.markdown(
        """
        Ask questions about demand forecasting, inventory,
        service levels, and supply chain planning.

        你可以询问需求预测、库存、服务水平以及供应链计划相关问题。
        """
    )

    # --------------------------------------------------------
    # INITIALIZE CHAT HISTORY
    # --------------------------------------------------------

    if "copilot_messages" not in st.session_state:

        st.session_state.copilot_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your Supply Chain Copilot. 👋\n\n"
                    "你好！我是你的供应链 AI 助手。\n\n"
                    "You can ask me questions such as:\n"
                    "- Which SKU has the highest safety stock?\n"
                    "- What is the reorder point for SKU-01?\n"
                    "- Which forecasting model performs best?\n"
                    "- What happens if the service level changes?\n\n"
                    "目前我处于 Demo Mode。下一阶段会接入 LLM。"
                )
            }
        ]

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.copilot_messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_question = st.chat_input(
        "Ask a supply chain question... / 输入你的供应链问题..."
    )

    if user_question:

        # ----------------------------------------------------
        # SAVE USER MESSAGE
        # ----------------------------------------------------

        st.session_state.copilot_messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_question
            )

        # ----------------------------------------------------
        # DEMO RESPONSE
        #
        # IMPORTANT:
        # This is intentionally NOT using an LLM yet.
        # ----------------------------------------------------

# ----------------------------------------------------
# CALL QWEN LLM
# ----------------------------------------------------

try:

    # Build conversation history
    messages = [
        {
            "role": "system",
            "content": """
 You are an AI Supply Chain Copilot.

 你的角色是一个专业的供应链 AI 助手。

 Language rules:
 1. If the user asks in Chinese, answer primarily in Chinese.
 2. If the user asks in English, answer primarily in English.
 3. If the user mixes Chinese and English, you may respond bilingually when useful.
 4. Use clear, professional supply chain terminology.
 5. Do not invent data from the user's forecasting or inventory system.
 6. If specific project data is required but no tool/data has been provided,
    clearly state that you currently cannot access that data.

 You can explain concepts such as:
 - Demand Forecasting
 - MAPE
 - RMSE
 - Safety Stock
 - Reorder Point
 - Order-Up-To Level
 - EOQ
 - Service Level
 - Inventory Cost
 - Supply Chain Planning

 You are currently in the LLM-only phase.
 You do NOT yet have access to the project's CSV data or calculation tools.
 """
        }
    ]

    # Add previous conversation history
    for message in st.session_state.copilot_messages:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3
    )

    assistant_response = response.choices[0].message.content

except Exception as e:

    assistant_response = (
        "⚠️ LLM 调用失败。\n\n"
        f"Error: `{str(e)}`"
    )

# ----------------------------------------------------
# SAVE ASSISTANT RESPONSE
# ----------------------------------------------------

st.session_state.copilot_messages.append(
    {
        "role": "assistant",
        "content": assistant_response
    }
)

with st.chat_message("assistant"):
    st.markdown(
        assistant_response
    )

