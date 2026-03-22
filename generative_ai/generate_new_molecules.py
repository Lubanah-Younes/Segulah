"""
generate_new_molecules.py
Generate new molecules using model trained on real data
"""

import pandas as pd
import numpy as np
import joblib
import random
import os

print("=" * 60)
print("Generating New Molecules with Real Data Model")
print("=" * 60)

# Load model
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, "results", "real_features_model.pkl")
model = joblib.load(model_path)

# Generate new molecules
np.random.seed(42)
n_new = 1000

print(f"Generating {n_new} new molecules...")

# Feature ranges (based on real data distribution)
feature_ranges = {
    'MolWt': (200, 600),
    'LogP': (-2, 5),
    'NumHDonors': (0, 5),
    'NumHAcceptors': (0, 10),
    'NumRotatableBonds': (0, 12),
    'NumAromaticRings': (0, 4),
    'TPSA': (20, 140),
    'FractionCSP3': (0, 1)
}

# Generate random features
new_features = []
for i in range(n_new):
    features = []
    for feat, (min_val, max_val) in feature_ranges.items():
        features.append(random.uniform(min_val, max_val))
    new_features.append(features)

feature_cols = list(feature_ranges.keys())
X_new = pd.DataFrame(new_features, columns=feature_cols)

# Predict
y_pred_log = model.predict(X_new)
y_pred_ic50 = 10 ** y_pred_log

# Add predictions
X_new['predicted_log_ic50'] = y_pred_log
X_new['predicted_ic50_nM'] = y_pred_ic50

# Sort by potency
X_new = X_new.sort_values('predicted_ic50_nM')

print(f"\n========== TOP 10 PREDICTED MOLECULES ==========")
print(X_new.head(10).to_string())

# Save results
output_path = os.path.join(project_root, "results", "new_molecules_real_model.csv")
X_new.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# Statistics
highly_potent = X_new[X_new['predicted_ic50_nM'] < 10]
very_potent = X_new[X_new['predicted_ic50_nM'] < 1]
extremely_potent = X_new[X_new['predicted_ic50_nM'] < 0.1]

print(f"\n========== SUMMARY ==========")
print(f"Total molecules generated: {n_new}")
print(f"Molecules with predicted IC50 < 10 nM: {len(highly_potent)}")
print(f"Molecules with predicted IC50 < 1 nM: {len(very_potent)}")
print(f"Molecules with predicted IC50 < 0.1 nM: {len(extremely_potent)}")

if len(very_potent) > 0:
    print(f"\n🚀 STRONGEST PREDICTED MOLECULE:")
    best = very_potent.iloc[0]
    print(f"   IC50 = {best['predicted_ic50_nM']:.4f} nM")
    print(f"   Features:")
    for col in feature_cols:
        print(f"     {col}: {best[col]:.2f}")

print("\n✅ Generation complete!")