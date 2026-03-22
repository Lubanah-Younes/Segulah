"""
improve_model.py
Improve model by reducing overfitting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Improving Model - Reducing Overfitting")
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Define models to test
models = {
    'Original (Complex)': RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    'Simplified 1': RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    'Simplified 2': RandomForestRegressor(
        n_estimators=50,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1
    ),
    'Very Simple': RandomForestRegressor(
        n_estimators=30,
        max_depth=3,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
}

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

print("\n========== TESTING MODELS ==========")

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Train and test
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    # Overfitting gap
    overfit_gap = train_r2 - cv_mean
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'CV Mean R²': cv_mean,
        'CV Std': cv_std,
        'Overfit Gap': overfit_gap
    })
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  CV Mean R²: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Overfit Gap: {overfit_gap:.4f}")
    
    if overfit_gap < 0.15:
        print(f"  ✅ GOOD: Low overfitting!")
    else:
        print(f"  ⚠️ Overfitting detected: gap = {overfit_gap:.4f}")

# Create results table
results_df = pd.DataFrame(results)
print("\n========== SUMMARY ==========")
print(results_df.to_string(index=False))

# Find best model based on CV R²
best_idx = results_df['CV Mean R²'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_cv = results_df.loc[best_idx, 'CV Mean R²']

print(f"\n🏆 Best model: {best_model_name}")
print(f"   Cross-validation R²: {best_cv:.4f}")

# Train best model on all data
best_model = models[best_model_name]
best_model.fit(X, y)
final_train_r2 = best_model.score(X, y)

print(f"\n========== FINAL MODEL ==========")
print(f"Model: {best_model_name}")
print(f"Parameters:")
for name, model in models.items():
    if name == best_model_name:
        print(f"  n_estimators: {model.n_estimators}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  min_samples_split: {model.min_samples_split}")
        print(f"  min_samples_leaf: {model.min_samples_leaf}")

print(f"\nFinal model R² (all data): {final_train_r2:.4f}")

# Save best model
final_model_path = os.path.join(project_root, "results", "improved_rf_model.pkl")
joblib.dump(best_model, final_model_path)
print(f"✅ Improved model saved to: {final_model_path}")

# Save results
results_df.to_csv(os.path.join(project_root, "results", "model_comparison.csv"), index=False)
print(f"✅ Comparison saved to: {os.path.join(project_root, 'results', 'model_comparison.csv')}")

# Plot comparison
plt.figure(figsize=(12, 6))
x = range(len(results_df))
width = 0.25

plt.bar([i - width for i in x], results_df['Train R²'], width, label='Train R²', alpha=0.8)
plt.bar(x, results_df['CV Mean R²'], width, label='CV R²', alpha=0.8)
plt.bar([i + width for i in x], results_df['Test R²'], width, label='Test R²', alpha=0.8)

plt.xlabel('Model')
plt.ylabel('R² Score')
plt.title('Model Comparison: Train vs CV vs Test')
plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(project_root, "results", "model_comparison.png"))
print(f"✅ Plot saved to: {os.path.join(project_root, 'results', 'model_comparison.png')}")

print("\n✅ Model improvement complete!")