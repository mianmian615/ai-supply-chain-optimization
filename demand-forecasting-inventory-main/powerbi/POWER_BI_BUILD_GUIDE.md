# Build the Demand Forecasting report in Power BI

1. **Load** `sales.csv`, `forecast_30d.csv`, `inventory_recommendations.csv`; add a `Date` table.
2. **Model:** relate `sales[date]` and `forecast_30d[date]` → `Date[Date]`; build a `SKU` dimension (one row per sku) and relate all three tables to it.
3. **Measures:** paste from `dax_measures.md`.
4. **Theme:** View → Themes → Browse → `demand_forecasting_theme.json`.
5. **Page 1 (Overview):** KPI cards (Units Sold, Reorder Units, backtest MAPE from `results.json`); a line of actual `Units Sold` with `Forecast Units` overlaid for a chosen SKU; a bar of avg units by weekday.
6. **Page 2 (Inventory):** a clustered bar of Safety Stock / Reorder Point / Order-Up-To by SKU; a reorder-recommendations table with conditional formatting on `reorder_point`.

`index.html` shows the exact target layout, rendered from the same computed metrics.