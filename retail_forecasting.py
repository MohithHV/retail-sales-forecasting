"""
Retail Sales Forecasting & Demand Analysis
--------------------------------------------
Forecasts daily unit sales per (store, item) using lag-based and
rolling-window features with an XGBoost regressor.

Dataset: Kaggle "Store Item Demand Forecasting Challenge"
913,000 daily records | 10 stores | 50 items | 2013-01-01 to 2017-12-31

NOTE: Run this with `pip install xgboost` first -- this is the real
script for the resume numbers. (A stand-in preview using sklearn's
HistGradientBoostingRegressor was used only to sanity-check the pipeline
in an environment where xgboost couldn't be installed -- see
preview_with_sklearn.py. The numbers that go on the resume should come
from THIS script.)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("train.csv", parse_dates=["date"])
print(f"Dataset: {df.shape[0]:,} rows | {df['store'].nunique()} stores | "
      f"{df['item'].nunique()} items | {df['date'].min().date()} to {df['date'].max().date()}")

# ---------------------------------------------------------
# 2. EDA -- trends and seasonality (identifies demand trends / temporal patterns)
# ---------------------------------------------------------
daily_total = df.groupby("date")["sales"].sum().reset_index()

plt.figure(figsize=(12, 4))
plt.plot(daily_total["date"], daily_total["sales"], linewidth=0.6)
plt.title("Total Daily Sales Across All Stores/Items (2013-2017)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("eda_daily_sales_trend.png", dpi=120)
plt.close()

df["month"] = df["date"].dt.month
monthly_avg = df.groupby("month")["sales"].mean()
plt.figure(figsize=(8, 4))
monthly_avg.plot(kind="bar")
plt.title("Average Sales by Month (Seasonality)")
plt.xlabel("Month")
plt.ylabel("Average Daily Sales per (store, item)")
plt.tight_layout()
plt.savefig("eda_monthly_seasonality.png", dpi=120)
plt.close()

df["dow"] = df["date"].dt.dayofweek
dow_avg = df.groupby("dow")["sales"].mean()
print(f"\nWeekday effect (0=Mon..6=Sun) - avg sales:\n{dow_avg.round(2)}")

yearly_avg = df.groupby(df["date"].dt.year)["sales"].mean()
print(f"\nYear-over-year average daily sales (growth trend):\n{yearly_avg.round(2)}")

# ---------------------------------------------------------
# 3. Feature engineering -- lag & rolling features (per store-item series)
# ---------------------------------------------------------
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

group = df.groupby(["store", "item"])["sales"]
df["lag_1"] = group.shift(1)          # yesterday's sales
df["lag_7"] = group.shift(7)          # same day last week
df["lag_14"] = group.shift(14)        # same day 2 weeks ago
df["rolling_mean_7"] = group.shift(1).rolling(7).mean()    # trailing 7-day avg
df["rolling_mean_28"] = group.shift(1).rolling(28).mean()  # trailing 28-day avg
df["rolling_std_7"] = group.shift(1).rolling(7).std()      # trailing 7-day volatility

df["day"] = df["date"].dt.day
df["year"] = df["date"].dt.year
df["is_weekend"] = (df["dow"] >= 5).astype(int)

df = df.dropna().reset_index(drop=True)  # drop rows where lag/rolling windows aren't full yet
print(f"\nRows after feature engineering (lag warm-up dropped): {df.shape[0]:,}")

FEATURES = ["store", "item", "month", "day", "dow", "year", "is_weekend",
            "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_28", "rolling_std_7"]
TARGET = "sales"

# ---------------------------------------------------------
# 4. Time-based train/test split (NOT random -- this is a time series;
#    a random split would leak future information into training)
# ---------------------------------------------------------
split_date = "2017-10-01"  # last ~3 months held out for testing
train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]
print(f"\nTrain: {train.shape[0]:,} rows (up to {split_date})")
print(f"Test:  {test.shape[0]:,} rows (from {split_date} onward)")

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ---------------------------------------------------------
# 5. Baseline model -- naive forecast (predict yesterday's value) for comparison
# ---------------------------------------------------------
baseline_pred = test["lag_1"]
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
print(f"\nBaseline (naive lag-1 forecast) - MAE: {baseline_mae:.3f}, RMSE: {baseline_rmse:.3f}")

# ---------------------------------------------------------
# 6. XGBoost model
# ---------------------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test.replace(0, np.nan))) * 100

print(f"\nXGBoost - MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%")
improvement = 100 * (1 - mae / baseline_mae)
print(f"Improvement over naive baseline: {improvement:.1f}%")

# ---------------------------------------------------------
# 7. Feature importance
# ---------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\nFeature importance:\n{importance.round(4)}")

plt.figure(figsize=(8, 5))
importance.plot(kind="barh")
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------
# 8. Plot actual vs predicted for one store-item series (visual sanity check)
# ---------------------------------------------------------
sample = test[(test["store"] == 1) & (test["item"] == 1)].copy()
sample["pred"] = model.predict(sample[FEATURES])
plt.figure(figsize=(12, 4))
plt.plot(sample["date"], sample["sales"], label="Actual", linewidth=1.2)
plt.plot(sample["date"], sample["pred"], label="Predicted", linewidth=1.2, linestyle="--")
plt.title("Actual vs Predicted Sales (Store 1, Item 1, Test Period)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
plt.close()

print("\nSaved: eda_daily_sales_trend.png, eda_monthly_seasonality.png, "
      "feature_importance.png, actual_vs_predicted.png")
