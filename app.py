import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Number of Orders Prediction", layout="wide")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("📦 Number of Orders Prediction (Demand Forecasting)")
st.write(
    "This app predicts the **number of customer orders** using pricing, marketing, user behavior, "
    "and time-based features."
)

st.sidebar.header("🔧 Input Features")

# -----------------------------
# Sidebar Inputs
# -----------------------------
price = st.sidebar.number_input("Price", min_value=0.0, value=500.0, step=10.0)
discount = st.sidebar.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

campaign_active = st.sidebar.selectbox("Campaign Active", [0, 1])

page_views = st.sidebar.number_input("Page Views", min_value=0.0, value=80.0, step=1.0)
add_to_cart = st.sidebar.number_input("Add to Cart", min_value=0.0, value=20.0, step=1.0)

day_name = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}
dayofweek = day_map[day_name]
is_weekend = 1 if dayofweek >= 5 else 0

month = st.sidebar.slider("Month", 1, 12, 12)
is_holiday = st.sidebar.selectbox("Holiday", [0, 1])

st.sidebar.subheader("📌 Lag / Rolling Features")
orders_lag_1 = st.sidebar.number_input("Orders Lag 1 Day", min_value=0.0, value=22.0, step=1.0)
orders_lag_7 = st.sidebar.number_input("Orders Lag 7 Days", min_value=0.0, value=18.0, step=1.0)
orders_lag_14 = st.sidebar.number_input("Orders Lag 14 Days", min_value=0.0, value=20.0, step=1.0)

rolling_mean_7 = st.sidebar.number_input("Rolling Mean 7 Days", min_value=0.0, value=21.0, step=1.0)
rolling_mean_14 = st.sidebar.number_input("Rolling Mean 14 Days", min_value=0.0, value=20.0, step=1.0)

# -----------------------------
# Prediction
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Input Summary")
    st.dataframe(X_input)

with col2:
    st.subheader("🎯 Prediction Output")

    if st.button("Predict Orders 🚀"):
        pred = model.predict(X_input)[0]
        st.success(f"📦 Predicted Orders: **{pred:.2f}**")

        st.metric("Predicted Orders", f"{pred:.2f}")

st.markdown("---")

# -----------------------------
# Optional: Show model details
# -----------------------------
st.subheader("📌 Model Info")
st.write("Best Model Used: **XGBoost** (saved as `models/best_model.pkl`)")

st.info("Tip: Try changing discount/campaign/page views and see how the prediction changes!")
