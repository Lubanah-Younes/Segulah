"""
xgboost_model.py
Train XGBoost model on real data for better predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Training XGBoost Model on REAL Data")
print("=" * 60)

# Load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "processed", "real_data_with_real_features.csv")
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} compounds")

# Features and target
feature_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'FractionCSP3']

X = df[feature_cols]
y = df['log_ic50']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n========== XGBOOST PERFORMANCE ==========")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Compare with Random Forest
print(f"\n========== COMPARISON ==========")
print(f"Random Forest Test R²: 0.5740")
print(f"XGBoost Test R²: {test_r2:.4f}")
if test_r2 > 0.574:
    print("✅ XGBoost performs better!")
else:
    print("Random Forest still better")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n========== FEATURE IMPORTANCE ==========")
for i, row in importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Save model
model_path = os.path.join(project_root, "results", "xgboost_model.pkl")
joblib.dump(xgb_model, model_path)
print(f"\n✅ XGBoost model saved to: {model_path}")

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log IC50")
plt.ylabel("Predicted log IC50")
plt.title(f"XGBoost Model (R² = {test_r2:.3f})")
plt.savefig(os.path.join(project_root, "results", "xgboost_predictions.png"))
print(f"✅ Plot saved")

print("\n✅ XGBoost training complete!")