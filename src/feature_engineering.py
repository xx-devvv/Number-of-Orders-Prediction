import pandas as pd

# Load dataset
df = pd.read_csv("data/orders_raw.csv")
df["date"] = pd.to_datetime(df["date"])

# Basic missing handling
df["product_id"] = df["product_id"].fillna(df["product_id"].mode()[0])
df["price"] = df["price"].fillna(df["price"].median())
df["discount"] = df["discount"].fillna(df["discount"].median())
df["page_views"] = df["page_views"].fillna(df["page_views"].median())
df["add_to_cart"] = df["add_to_cart"].fillna(df["add_to_cart"].median())

# Sort values (IMPORTANT for lag features)
df = df.sort_values(["product_id", "date"])

# ---------------------------
# Time-based features
# ---------------------------
df["dayofweek"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# ---------------------------
# Holiday feature (simple version)
# ---------------------------
# You can treat some fixed dates as holidays
holiday_dates = [
    "2023-01-26", "2023-08-15", "2023-10-02", "2023-12-25",
    "2024-01-26", "2024-08-15", "2024-10-02", "2024-12-25"
]
holiday_dates = pd.to_datetime(holiday_dates)

df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)

# ---------------------------
# Lag features (per product)
# ---------------------------
df["orders_lag_1"] = df.groupby("product_id")["orders"].shift(1)
df["orders_lag_7"] = df.groupby("product_id")["orders"].shift(7)
df["orders_lag_14"] = df.groupby("product_id")["orders"].shift(14)

# ---------------------------
# Rolling mean features (per product)
# ---------------------------
df["rolling_mean_7"] = df.groupby("product_id")["orders"].shift(1).rolling(7).mean()
df["rolling_mean_14"] = df.groupby("product_id")["orders"].shift(1).rolling(14).mean()

# Drop rows with NaN created due to lag/rolling
df = df.dropna()

# Save final dataset
df.to_csv("data/final_orders_features.csv", index=False)

print("Saved: data/final_orders_features.csv")
print("Final shape:", df.shape)
print(df.head())
