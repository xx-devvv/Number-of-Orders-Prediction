import joblib
import pandas as pd

# -----------------------------
# Helper functions (safe input)
# -----------------------------
def get_float(prompt, default):
    val = input(f"{prompt} (default={default}): ").strip()
    return float(val) if val != "" else float(default)

def get_int(prompt, default):
    val = input(f"{prompt} (default={default}): ").strip()
    return int(val) if val != "" else int(default)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/best_model.pkl")
print("✅ Model Loaded: models/best_model.pkl")

# -----------------------------
# Take user input
# -----------------------------
price = get_float("Enter price", 500)
discount = get_float("Enter discount (%)", 10)
campaign_active = get_int("Campaign active? (0/1)", 1)

page_views = get_float("Enter page views", 80)
add_to_cart = get_float("Enter add to cart count", 20)

dayofweek = get_int("Day of week (0=Mon ... 6=Sun)", 5)
month = get_int("Month (1-12)", 12)

is_weekend = 1 if dayofweek >= 5 else 0
is_holiday = get_int("Holiday? (0/1)", 0)

orders_lag_1 = get_float("Orders lag 1 day", 22)
orders_lag_7 = get_float("Orders lag 7 days", 18)
orders_lag_14 = get_float("Orders lag 14 days", 20)

rolling_mean_7 = get_float("Rolling mean 7 days", 21)
rolling_mean_14 = get_float("Rolling mean 14 days", 20)

# -----------------------------
# Create input dataframe
# (must match training features)
# -----------------------------
X_input = pd.DataFrame([{
    "price": price,
    "discount": discount,
    "campaign_active": campaign_active,
    "page_views": page_views,
    "add_to_cart": add_to_cart,
    "dayofweek": dayofweek,
    "month": month,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday,
    "orders_lag_1": orders_lag_1,
    "orders_lag_7": orders_lag_7,
    "orders_lag_14": orders_lag_14,
    "rolling_mean_7": rolling_mean_7,
    "rolling_mean_14": rolling_mean_14
}])

# -----------------------------
# Predict
# -----------------------------
pred = model.predict(X_input)[0]
print(f"\n📦 Predicted Orders: {pred:.2f}")
