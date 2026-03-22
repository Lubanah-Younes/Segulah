"""
train_model.py
Train machine learning model to predict compound activity
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

print("=" * 50)
print("AI Model Training")
print("=" * 50)

# Load data
print("Loading data...")
df = pd.read_csv("../ai_data.csv")
print(f"Loaded {len(df)} samples")

# Split features and target
feature_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 
                'FractionCSP3', 'NumHeteroatoms', 'NumHeavyAtoms']

X = df[feature_cols]
y = df['log_ic50']

print(f"Features: {len(feature_cols)}")
print(f"Target: log_ic50")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n========== MODEL PERFORMANCE ==========")
print(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n========== FEATURE IMPORTANCE ==========")
for i, row in feature_importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Save model
joblib.dump(model, "../ai_model.pkl")
print(f"\n✅ Model saved to: ../ai_model.pkl")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log IC50")
plt.ylabel("Predicted log IC50")
plt.title(f"Model Predictions (R² = {test_r2:.3f})")
plt.savefig("../ai_predictions.png")
print(f"✅ Plot saved to: ../ai_predictions.png")

print("\n✅ Training complete!")