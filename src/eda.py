import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/orders_raw.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

print("HEAD:\n", df.head())
print("\nINFO:\n")
print(df.info())
print("\nMISSING VALUES:\n", df.isnull().sum())

# Minimal missing value handling for EDA
df["product_id"] = df["product_id"].fillna(df["product_id"].mode()[0])
df["price"] = df["price"].fillna(df["price"].median())
df["discount"] = df["discount"].fillna(df["discount"].median())
df["page_views"] = df["page_views"].fillna(df["page_views"].median())

# -----------------------------
# (A) Total Orders Over Time
# -----------------------------
daily_orders = df.groupby("date")["orders"].sum()

plt.figure(figsize=(12, 5))
plt.plot(daily_orders.index, daily_orders.values)
plt.title("Total Orders Over Time")
plt.xlabel("Date")
plt.ylabel("Total Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (B) Avg Orders by Day of Week
# -----------------------------
df["day_name"] = df["date"].dt.day_name()

dow_orders = df.groupby("day_name")["orders"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

plt.figure(figsize=(10, 4))
dow_orders.plot(kind="bar")
plt.title("Average Orders by Day of Week")
plt.xlabel("Day")
plt.ylabel("Avg Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (C) Avg Orders by Month
# -----------------------------
df["month"] = df["date"].dt.month
month_orders = df.groupby("month")["orders"].mean()

plt.figure(figsize=(10, 4))
plt.plot(month_orders.index, month_orders.values, marker="o")
plt.title("Average Orders by Month")
plt.xlabel("Month")
plt.ylabel("Avg Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (D) Campaign Impact
# -----------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="campaign_active", y="orders", data=df)
plt.title("Orders vs Campaign Active")
plt.xlabel("Campaign Active (0=No, 1=Yes)")
plt.ylabel("Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (E) Discount vs Orders
# -----------------------------
plt.figure(figsize=(7, 4))
sns.scatterplot(x="discount", y="orders", data=df, alpha=0.4)
plt.title("Discount vs Orders")
plt.xlabel("Discount (%)")
plt.ylabel("Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (F) Correlation Heatmap
# -----------------------------
plt.figure(figsize=(8, 5))
sns.heatmap(
    df[["price", "discount", "campaign_active", "page_views", "add_to_cart", "orders"]].corr(),
    annot=True
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
