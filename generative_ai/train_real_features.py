"""
train_real_features.py
Train AI model on real data with real features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Training AI Model on REAL Data with REAL Features")
print("=" * 60)

# Get the project root (L:/drug_discovery)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "processed", "real_data_with_real_features.csv")

print(f"Loading data from: {data_path}")

# Load real data
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} compounds with REAL features")

# Feature columns
feature_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 'FractionCSP3']

# Target
X = df[feature_cols]
y = df['log_ic50']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target: log_ic50 (log10 of REAL IC50)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train model
print("\nTraining Random Forest model...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n========== MODEL PERFORMANCE ==========")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n========== FEATURE IMPORTANCE ==========")
for i, row in importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Save model
model_path = os.path.join(project_root, "results", "real_features_model.pkl")
joblib.dump(model, model_path)
print(f"\n✅ Model saved to: {model_path}")

# Plot predictions
plot_path = os.path.join(project_root, "results", "real_features_predictions.png")
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual log IC50")
plt.ylabel("Predicted log IC50")
plt.title(f"Real Features Model (R² = {test_r2:.3f})")
plt.savefig(plot_path)
print(f"✅ Plot saved to: {plot_path}")

# Save test predictions
results_path = os.path.join(project_root, "results", "test_predictions.csv")
results_df = pd.DataFrame({
    'actual_log_ic50': y_test,
    'predicted_log_ic50': y_pred_test,
    'actual_ic50_nM': 10 ** y_test,
    'predicted_ic50_nM': 10 ** y_pred_test
})
results_df.to_csv(results_path, index=False)
print(f"✅ Test predictions saved to: {results_path}")

print("\n✅ Training complete!")