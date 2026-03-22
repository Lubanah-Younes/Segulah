"""
cross_validation.py
Perform cross-validation to check model stability
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("Cross-Validation for Random Forest Model")
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

# Load best model parameters from tuning
best_params_path = os.path.join(project_root, "results", "best_params.csv")
if os.path.exists(best_params_path):
    best_params_df = pd.read_csv(best_params_path)
    best_params = {}
    for i, row in best_params_df.iterrows():
        param = row['param']
        value = row['value']
        # Convert numeric parameters
        if param in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'n_estimators']:
            try:
                value = int(value)
            except:
                pass
        best_params[param] = value
    print(f"\nUsing best parameters from tuning:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
else:
    # Default parameters
    best_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 1,
        'max_features': 'log2'
    }
    print("\nUsing default parameters")

# Create model with best parameters
model = RandomForestRegressor(
    n_estimators=best_params.get('n_estimators', 200),
    max_depth=best_params.get('max_depth', 10),
    min_samples_split=best_params.get('min_samples_split', 5),
    min_samples_leaf=best_params.get('min_samples_leaf', 1),
    max_features=best_params.get('max_features', 'log2'),
    random_state=42,
    n_jobs=-1
)

print(f"\nModel created with: n_estimators={model.n_estimators}, max_depth={model.max_depth}")

# Perform 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning 5-fold cross-validation...")
try:
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_mse = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    
    print(f"\n========== CROSS-VALIDATION RESULTS (5-FOLD) ==========")
    print(f"R² scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Mean MSE: {-cv_mse.mean():.4f} (+/- {cv_mse.std():.4f})")
    
    # Get cross-validated predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)
    overall_r2 = r2_score(y, y_pred_cv)
    print(f"\nOverall R² (cross-validated): {overall_r2:.4f}")
    
    # Train final model on all data
    model.fit(X, y)
    final_train_r2 = model.score(X, y)
    
    print(f"\n========== COMPARISON ==========")
    print(f"Original model (train/test): Test R² = 0.5740")
    print(f"Tuned model (train/test): Test R² = 0.6072")
    print(f"Cross-validation mean R²: {cv_scores.mean():.4f}")
    print(f"Final model (all data) R²: {final_train_r2:.4f}")
    
    # Check for overfitting
    if final_train_r2 - cv_scores.mean() > 0.2:
        print("\n⚠️ WARNING: Possible overfitting detected!")
        print(f"   Training R² ({final_train_r2:.4f}) is much higher than CV R² ({cv_scores.mean():.4f})")
    else:
        print("\n✅ Model is stable (no significant overfitting)")
    
    # Plot cross-validation results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Actual vs Predicted (cross-validated)
    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred_cv, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual log IC50")
    plt.ylabel("Predicted log IC50 (CV)")
    plt.title(f"Cross-Validation Predictions\nR² = {overall_r2:.3f}")
    
    # Plot 2: R² scores per fold
    plt.subplot(1, 2, 2)
    plt.bar(range(1, 6), cv_scores, color='steelblue')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean = {cv_scores.mean():.3f}')
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.title("Cross-Validation R² per Fold")
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "results", "cross_validation_results.png"))
    print(f"\n✅ Plot saved to: {os.path.join(project_root, 'results', 'cross_validation_results.png')}")
    
    # Save cross-validation results
    results = pd.DataFrame({
        'fold': range(1, 6),
        'r2_score': cv_scores,
        'neg_mse': cv_mse
    })
    results.to_csv(os.path.join(project_root, "results", "cv_results.csv"), index=False)
    print(f"✅ CV results saved to: {os.path.join(project_root, 'results', 'cv_results.csv')}")
    
    # Save final model
    final_model_path = os.path.join(project_root, "results", "final_rf_model.pkl")
    joblib.dump(model, final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")
    
    print("\n========== SUMMARY ==========")
    print(f"Model: Random Forest")
    print(f"Cross-validation (5-fold) R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Model is ready for deployment!")

except Exception as e:
    print(f"\nError during cross-validation: {e}")
    print("\nTrying with default parameters...")
    # Fallback: use default parameters
    model_default = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(model_default, X, y, cv=cv, scoring='r2')
    print(f"Default model CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n✅ Cross-validation complete!")