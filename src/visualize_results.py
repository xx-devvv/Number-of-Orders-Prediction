import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/final_orders_features.csv")

# Features / target
X = df.drop(columns=["orders", "date", "product_id"])
y = df["orders"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load best model
model = joblib.load("models/best_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# -----------------------------
# (A) Actual vs Predicted Scatter
# -----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4)
plt.title("Actual vs Predicted Orders")
plt.xlabel("Actual Orders")
plt.ylabel("Predicted Orders")
plt.tight_layout()
plt.show()

# -----------------------------
# (B) Actual vs Predicted Line (first 100 samples)
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title("Actual vs Predicted Orders (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Orders")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# (C) Feature Importance (XGBoost)
# -----------------------------
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = X.columns

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
else:
    print("Model does not support feature_importances_")
