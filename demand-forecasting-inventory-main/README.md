# 📦 AI 驱动的供应链需求预测与库存优化系统

# AI-Driven Supply Chain Demand Forecasting & Inventory Optimization

一个面向供应链需求预测与库存决策的端到端数据分析与 AI 决策支持系统。

An end-to-end supply chain analytics and AI decision-support system for demand forecasting, inventory optimization, and AI-assisted supply chain planning.

---

## 🇨🇳 项目简介 | Project Overview

本项目从历史销售数据出发，构建了一套从：

**数据分析 → 需求预测 → 模型评估 → 库存优化 → 成本分析 → Dashboard → AI Supply Chain Copilot**

的完整供应链分析流程。

系统主要解决以下问题：

* 未来 30 天需求是多少？
* 不同 SKU 应该使用什么预测模型？
* 什么时候应该补货？
* 应该保留多少安全库存？
* 服务水平提高后，库存成本如何变化？
* AI 能否直接读取项目中的实际数据并回答供应链问题？

项目最终将传统的数据分析 / 预测流程扩展为一个可以交互使用的供应链决策支持应用。

### English

This project builds an end-to-end supply chain decision-support workflow from historical sales data:

**Data Analysis → Demand Forecasting → Model Evaluation → Inventory Optimization → Cost Analysis → Dashboard → AI Supply Chain Copilot**

The system focuses on practical supply chain questions such as:

* What will demand look like over the next 30 days?
* Which forecasting model performs better for each SKU?
* When should inventory be reordered?
* How much safety stock should be maintained?
* How does service level affect inventory cost?
* Can an AI assistant answer questions using actual project data?

---

# 🏗️ 系统架构 | System Architecture

```text
Historical Sales Data
历史销售数据
        │
        ▼
Data Analysis
数据分析
        │
        ▼
Demand Forecasting
需求预测
 ├── Seasonal Baseline
 ├── Seasonal + Promotion
 └── Linear Regression
        │
        ▼
Backtesting & Model Comparison
回测与模型比较
 ├── MAPE
 └── RMSE
        │
        ▼
Inventory Optimization
库存优化
 ├── Safety Stock
 ├── Reorder Point
 ├── Order-Up-To Level
 └── EOQ
        │
        ▼
Inventory Cost Analysis
库存成本分析
        │
        ▼
Streamlit Dashboard
交互式 Dashboard
        │
        ▼
AI Supply Chain Copilot
AI 供应链助手
        │
        ▼
Qwen LLM + Tool Calling
```

---

# 📊 数据集 | Dataset

项目使用模拟零售销售数据进行供应链建模与分析。

The project uses a synthetic retail sales dataset for forecasting and inventory optimization.

### 数据规模 | Dataset Size

* **6 SKUs**
* **2 years of daily data**
* **4,380 records**
* Product category information
* Promotion information

主要字段：

```text
date
sku
category
units_sold
promo
```

数据包含趋势、周期性需求以及促销影响，用于模拟实际供应链中的需求波动。

The dataset contains trend, seasonal patterns, and promotion effects to simulate realistic demand behavior.

---

# 🔮 需求预测 | Demand Forecasting

项目实现并比较了三种需求预测方法。

The project implements and compares three forecasting approaches.

### 1. Seasonal Baseline

基于历史需求中的周期性模式建立基准预测模型。

A baseline forecasting model that captures recurring seasonal demand patterns.

### 2. Seasonal + Promotion

在季节性预测的基础上加入促销信息。

An enhanced seasonal model that incorporates promotion information.

### 3. Linear Regression

使用时间特征、星期、月份和促销信息进行需求预测。

A regression model using time-related and promotion-related features.

---

# 📈 模型回测 | Model Backtesting

为了避免未来数据泄露，项目采用基于时间顺序的 Backtesting，而不是随机划分训练集和测试集。

The project uses time-based backtesting instead of random train/test splitting to better reflect real-world forecasting.

当前实验使用最后 **60 天**作为 Holdout Period。

The latest **60 days** are used as the holdout period.

评估指标：

* **MAPE — Mean Absolute Percentage Error**
* **RMSE — Root Mean Squared Error**

---

# 📊 模型比较结果 | Model Comparison

当前数据集上的实验结果：

| 模型 Model             |   MAPE |  RMSE |
| -------------------- | -----: | ----: |
| Seasonal Baseline    |  8.71% |  9.62 |
| Seasonal + Promotion |  8.21% |  7.77 |
| Linear Regression    | 15.32% | 13.60 |

这些结果来自当前项目中的模拟数据和回测设置。

These results are based on the current synthetic dataset and backtesting configuration.

同时，项目进一步进行了 **SKU-level model comparison**，因为不同 SKU 可能具有不同的需求模式。

The project also evaluates model performance at the SKU level because different products may have different demand patterns.

---

# 📦 库存优化 | Inventory Optimization

需求预测结果进一步用于库存决策。

Forecasting results are transformed into inventory planning decisions.

项目包括：

* Safety Stock 安全库存
* Reorder Point 再订货点
* Order-Up-To Level 订货上限
* EOQ 经济订货量
* Service Level 服务水平分析

---

## 🛡️ 安全库存 | Safety Stock

安全库存用于应对需求波动和提前期内的不确定性。

Safety stock is used to protect against demand variability and uncertainty during lead time.

项目分析：

* 90% Service Level
* 95% Service Level
* 99% Service Level

一般情况下：

```text
Higher Service Level
        ↓
Higher Safety Stock
        ↓
Higher Holding Cost
        ↓
Lower Expected Stockout Risk
```

服务水平不是越高越好，而是需要在库存成本和缺货风险之间进行权衡。

The goal is to understand the trade-off between inventory cost and stockout risk.

---

## 🔄 再订货点 | Reorder Point

再订货点用于判断什么时候需要补货。

The reorder point determines when inventory should be replenished.

核心逻辑：

```text
Reorder Point
= Lead-Time Demand
+ Safety Stock
```

其中：

* Lead-Time Demand：提前期内的预期需求
* Safety Stock：用于应对需求不确定性的库存缓冲

---

## 📦 EOQ

项目同时计算 Economic Order Quantity（EOQ），用于分析订货成本与库存持有成本之间的关系。

EOQ is also calculated to analyze the trade-off between ordering cost and inventory holding cost.

EOQ 与当前的周期性库存补货策略分别进行分析，避免将两个不同的库存决策逻辑混淆。

EOQ analysis is kept separate from the current periodic-review inventory policy.

---

# 💰 库存成本分析 | Inventory Cost Analysis

项目将库存成本拆分为：

The inventory cost analysis includes:

* Holding Cost
* Ordering Cost
* Stockout Cost

并比较不同 Service Level 下的库存成本变化。

The system compares inventory cost under different service levels.

核心思想：

```text
Service Level
      ↓
Safety Stock
      ↓
Average Inventory
      ↓
Holding Cost

Service Level
      ↓
Expected Stockout
      ↓
Stockout Cost
```

因此，库存优化并不是简单地追求最高 Service Level，而是在服务水平与库存成本之间寻找合理的决策方案。

The objective is not simply to maximize service level, but to understand the trade-off between service performance and inventory cost.

---

# 🖥️ Streamlit Dashboard

项目使用 Streamlit 构建交互式 Web Dashboard。

The project uses Streamlit to build an interactive web dashboard.

主要页面包括：

### Overview | 总览

展示项目整体 KPI 和预测信息。

### Demand Forecast | 需求预测

查看不同 SKU 的未来需求预测结果。

### Inventory Optimization | 库存优化

查看：

* Average Daily Demand
* Demand Standard Deviation
* Safety Stock
* Reorder Point
* Order-Up-To Level
* EOQ
* Inventory Cost

### Model Comparison | 模型比较

比较不同需求预测模型的表现。

### AI Supply Chain Copilot | AI 供应链助手

通过 Qwen LLM + Tool Calling，让 AI 助手能够访问项目实际数据并回答供应链问题。

---

# 🤖 AI Supply Chain Copilot

项目集成了 **Qwen LLM**，并通过 Alibaba Cloud Model Studio 的 OpenAI-compatible API 接口进行调用。

The project integrates **Qwen LLM** through Alibaba Cloud Model Studio's OpenAI-compatible API interface.

用户可以直接提出类似的问题：

```text
SKU-01 的 Reorder Point 是多少？

Which forecasting model performs best for SKU-03?

SKU-02 的 Safety Stock 是多少？

Why does a higher service level require more inventory?
```

---

# 🔧 Tool Calling

项目没有让 LLM 单独“猜测”项目数据，而是实现了 Tool Calling。

Instead of relying only on the LLM's internal knowledge, the Copilot can call Python functions to retrieve actual project data.

基本流程：

```text
User Question
用户问题
      │
      ▼
Qwen LLM
      │
      │ decides whether a tool is needed
      ▼
Python Tool
      │
      ▼
Project CSV Data
      │
      ▼
Actual Inventory / Model Data
      │
      ▼
Qwen LLM
      │
      ▼
Final Answer
最终回答
```

目前实现的主要工具：

```text
get_inventory_data()
get_model_performance()
```

例如，当用户询问：

> What is the reorder point for SKU-01?

Qwen 可以判断需要调用库存数据工具，然后读取项目中的实际 SKU 数据，再生成最终回答。

This allows the Copilot to provide data-grounded answers instead of generating unsupported values.

---

# 🛠️ 技术栈 | Tech Stack

### Programming

* Python
* Pandas
* NumPy
* Matplotlib

### Forecasting

* Time Series Forecasting
* Seasonal Modeling
* Linear Regression
* Backtesting
* MAPE
* RMSE

### Supply Chain

* Safety Stock
* Reorder Point
* Order-Up-To Level
* EOQ
* Service Level Analysis
* Inventory Cost Analysis

### Application

* Streamlit

### AI

* Qwen
* Alibaba Cloud Model Studio
* OpenAI-compatible API
* Function / Tool Calling

### Development

* Git
* GitHub
* PyCharm

---

# 📁 项目结构 | Project Structure

```text
ai-supply-chain-optimization/
│
├── app.py
├── ai_tools.py
├── forecast.py
├── generate_data.py
├── test_qwen.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sales.csv
│
└── outputs/
    ├── forecast_30d.csv
    ├── inventory_optimization.csv
    ├── model_comparison.csv
    ├── model_comparison_by_sku.csv
    ├── service_level_analysis.csv
    ├── results.json
    ├── eoq_by_sku.png
    ├── inventory_cost_components.png
    └── service_level_vs_cost.png
```

---

# ▶️ 本地运行 | Run Locally

## 1. Clone Repository

```bash
git clone https://github.com/mianmian615/ai-supply-chain-optimization.git
cd ai-supply-chain-optimization
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure API Key

项目通过环境变量读取 Qwen API Key：

```text
DASHSCOPE_API_KEY
```

Windows PowerShell 示例：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

**不要将真实 API Key 写入 Python 文件或提交到 GitHub。**

**Never hard-code the API key into source code or commit it to GitHub.**

## 5. Run Dashboard

```bash
streamlit run app.py
```

然后在浏览器打开 Streamlit 提供的本地地址。

---

# 🔐 安全性 | Security

项目使用环境变量管理 API Key。

Sensitive credentials are intentionally excluded from the repository.

`.gitignore` 会排除：

```text
.env
.streamlit/secrets.toml
venv/
.venv/
.idea/
__pycache__/
```

部署到云端时，API Key 应通过云平台的 Secrets / Environment Variables 配置，而不是写入代码。

For cloud deployment, the API key should be configured through the platform's secret-management system.

---

# 🚀 项目开发过程 | Development Process

本项目是在理解并复现已有需求预测与库存优化流程的基础上进行扩展。

The project was developed by reproducing and understanding an existing demand forecasting and inventory optimization workflow, and then extending it.

主要扩展包括：

1. 复现原始需求预测与库存优化流程
2. 分析时间序列预测与库存优化逻辑
3. 增加 Promotion-aware Forecasting
4. 增加多个 Forecasting Model 的比较
5. 增加 SKU-level Model Comparison
6. 增加 Service Level Analysis
7. 增加 Inventory Cost Analysis
8. 增加 EOQ 分析
9. 使用 Streamlit 构建交互式 Dashboard
10. 集成 Qwen LLM
11. 实现 Tool Calling
12. 让 AI 助手能够读取项目实际数据
13. 准备云端部署

---

# 🔮 后续计划 | Future Improvements

未来可以进一步扩展：

* More advanced forecasting models
* Automated model selection by SKU
* Multi-echelon inventory optimization
* Production capacity constraints
* Warehouse capacity constraints
* Transportation cost optimization
* Supplier lead-time uncertainty
* Promotion uplift modeling
* Automated replenishment recommendations
* FastAPI backend
* More Supply Chain Copilot tools
* What-if scenario simulation

---

# 👤 Author

**Mianmian**

GitHub:
https://github.com/mianmian615

---

# 📄 License

This project is intended for educational, portfolio, and interview demonstration purposes.
