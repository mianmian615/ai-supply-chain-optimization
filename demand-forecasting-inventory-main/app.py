
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

# ============================================================
# AI SUPPLY CHAIN COPILOT
# ============================================================

elif page == "AI Supply Chain Copilot / AI供应链助手":

    from ai_tools import (
        get_inventory_data,
        get_model_performance
    )

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
                    "现在我可以读取项目中的预测和库存分析数据。\n\n"
                    "例如你可以问：\n"
                    "- SKU-01 的补货点是多少？\n"
                    "- SKU-01 的安全库存是多少？\n"
                    "- SKU-01 适合什么预测模型？\n"
                    "- Which SKU has the highest safety stock?\n"
                    "- What is the EOQ for SKU-01?"
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
        # SYSTEM PROMPT
        # ----------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": """
You are an AI Supply Chain Copilot.

你的角色是一个专业的供应链 AI 助手。

Language rules:
1. If the user asks in Chinese, answer primarily in Chinese.
2. If the user asks in English, answer primarily in English.
3. If the user mixes Chinese and English, respond bilingually when useful.
4. Use clear and professional supply chain terminology.

IMPORTANT DATA RULES:

You have access to project data through Python tools.

When the user asks about specific SKU data,
DO NOT guess or invent values.

Use the available tools to retrieve the actual project data.

Available project data includes:

- Demand forecasting results
- Forecasting model performance
- Safety stock
- Reorder point
- Order-up-to level
- EOQ
- Average inventory
- Inventory cost
- Service level
- Demand statistics

When answering numerical questions about the project,
use the values returned by the tools.

For conceptual questions such as
"What is safety stock?",
you may answer directly without using a tool.

If the requested project data cannot be found,
clearly state that the data is unavailable.

Do not fabricate project data.
"""
            }
        ]

        # ----------------------------------------------------
        # ADD CONVERSATION HISTORY
        # ----------------------------------------------------

        for message in st.session_state.copilot_messages:

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"]
                }
            )

        # ----------------------------------------------------
        # DEFINE TOOLS
        # ----------------------------------------------------

        tools = [

            {
                "type": "function",
                "function": {
                    "name": "get_inventory_data",
                    "description": (
                        "Get inventory optimization data for a specific SKU, "
                        "including safety stock, reorder point, "
                        "order-up-to level, EOQ, inventory cost, "
                        "service level, and forecast model."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": (
                                    "SKU identifier, for example SKU-01"
                                )
                            }
                        },
                        "required": ["sku"]
                    }
                }
            },

            {
                "type": "function",
                "function": {
                    "name": "get_model_performance",
                    "description": (
                        "Get forecasting model MAPE performance "
                        "for a specific SKU and identify the model "
                        "with the lowest MAPE."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": (
                                    "SKU identifier, for example SKU-01"
                                )
                            }
                        },
                        "required": ["sku"]
                    }
                }
            }
        ]

        # ----------------------------------------------------
        # FIRST QWEN CALL
        # ----------------------------------------------------

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3
            )

            assistant_message = response.choices[0].message

            # ------------------------------------------------
            # CHECK WHETHER QWEN WANTS TO CALL A TOOL
            # ------------------------------------------------

            if assistant_message.tool_calls:

                messages.append(
                    assistant_message
                )

                # --------------------------------------------
                # EXECUTE EACH TOOL CALL
                # --------------------------------------------

                for tool_call in assistant_message.tool_calls:

                    function_name = tool_call.function.name

                    import json

                    arguments = json.loads(
                        tool_call.function.arguments
                    )

                    if function_name == "get_inventory_data":

                        tool_result = get_inventory_data(
                            arguments["sku"]
                        )

                    elif function_name == "get_model_performance":

                        tool_result = get_model_performance(
                            arguments["sku"]
                        )

                    else:

                        tool_result = {
                            "success": False,
                            "message": "Unknown tool."
                        }

                    # ----------------------------------------
                    # SEND TOOL RESULT BACK TO QWEN
                    # ----------------------------------------

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                tool_result,
                                ensure_ascii=False
                            )
                        }
                    )

                # --------------------------------------------
                # SECOND QWEN CALL
                # --------------------------------------------

                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3
                )

                assistant_response = (
                    final_response
                    .choices[0]
                    .message
                    .content
                )

            else:

                # Qwen answered directly
                assistant_response = (
                    assistant_message.content
                )

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