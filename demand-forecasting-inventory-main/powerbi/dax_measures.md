# DAX measures — Demand Forecasting & Inventory

Tables: `sales` (`date, sku, category, units_sold, promo`), `forecast_30d` (`sku, date, forecast_units`),
`inventory_recommendations` (`sku, category, avg_daily_demand, demand_std, lead_time_demand,
safety_stock, reorder_point, order_up_to_level`), `Date`.

## Demand & forecast
```DAX
Units Sold       = SUM ( sales[units_sold] )
Avg Daily Demand = AVERAGEX ( VALUES ( 'Date'[Date] ), [Units Sold] )
Forecast Units   = SUM ( forecast_30d[forecast_units] )
Forecast 30d     = CALCULATE ( [Forecast Units], forecast_30d[date] <= MAX('Date'[Date]) + 30 )
```

## Inventory policy
```DAX
Safety Stock   = SUM ( inventory_recommendations[safety_stock] )
Reorder Point  = SUM ( inventory_recommendations[reorder_point] )
Order Up To    = SUM ( inventory_recommendations[order_up_to_level] )
Reorder Units  = SUMX ( inventory_recommendations,
                        MAX ( inventory_recommendations[order_up_to_level] - inventory_recommendations[reorder_point], 0 ) )
```
Add a `Weekday = FORMAT ( sales[date], "ddd" )` column to expose the weekly seasonality.