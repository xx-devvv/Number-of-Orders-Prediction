import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import joblib
import os

# Load final dataset
df = pd.read_csv("data/final_orders_features.csv")

# Drop non-numeric columns (date + product_id)
X = df.drop(columns=["orders", "date", "product_id"])
y = df["orders"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = []

# Train + Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append((name, rmse, mae, r2))

    print(f"\n{name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R2  : {r2:.4f}")

# Pick best model by RMSE
best_model = sorted(results, key=lambda x: x[1])[0][0]
print("\nBest Model (Lowest RMSE):", best_model)

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(models[best_model], "models/best_model.pkl")
print("Saved model to: models/best_model.pkl")
