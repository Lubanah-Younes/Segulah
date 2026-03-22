"""
hyperparameter_tuning.py
Optimize Random Forest hyperparameters for better R²
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import joblib
import os

print("=" * 60)
print("Hyperparameter Tuning for Random Forest")
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

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

print("\nParameter grid:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# Grid search
print("\nRunning Grid Search (this may take a few minutes)...")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("\n========== BEST PARAMETERS ==========")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

# Best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\n========== TUNED MODEL PERFORMANCE ==========")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Compare with original
original_test_r2 = 0.5740
improvement = test_r2 - original_test_r2
print(f"\n========== COMPARISON ==========")
print(f"Original Random Forest Test R²: {original_test_r2:.4f}")
print(f"Tuned Random Forest Test R²: {test_r2:.4f}")
if improvement > 0:
    print(f"✅ Improvement: +{improvement:.4f}")
else:
    print(f"⚠️ No improvement: {improvement:.4f}")

# Save best model
model_path = os.path.join(project_root, "results", "tuned_rf_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n✅ Tuned model saved to: {model_path}")

# Save results
results = pd.DataFrame({
    'param': list(grid_search.best_params_.keys()),
    'value': list(grid_search.best_params_.values())
})
results.to_csv(os.path.join(project_root, "results", "best_params.csv"), index=False)
print(f"✅ Best parameters saved")

# Feature importance from best model
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n========== FEATURE IMPORTANCE (TUNED) ==========")
for i, row in importance.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

print("\n✅ Hyperparameter tuning complete!")