"""
generate_molecules.py
Generate new molecules and predict their activity
"""

import pandas as pd
import numpy as np
import joblib
import random

print("=" * 50)
print("Generating New Molecules")
print("=" * 50)

# Load trained model
model = joblib.load("../ai_model.pkl")

# Generate random new molecules (simulated features)
np.random.seed(123)
n_new_molecules = 1000

print(f"Generating {n_new_molecules} new molecules...")

# Feature ranges based on training data
feature_ranges = {
    'MolWt': (200, 600),
    'LogP': (-2, 5),
    'NumHDonors': (0, 5),
    'NumHAcceptors': (0, 10),
    'NumRotatableBonds': (0, 10),
    'NumAromaticRings': (0, 4),
    'TPSA': (20, 140),
    'FractionCSP3': (0, 1),
    'NumHeteroatoms': (0, 8),
    'NumHeavyAtoms': (10, 40)
}

# Generate random features within ranges
new_features = []
for i in range(n_new_molecules):
    features = []
    for feat, (min_val, max_val) in feature_ranges.items():
        features.append(random.uniform(min_val, max_val))
    new_features.append(features)

feature_cols = list(feature_ranges.keys())
X_new = pd.DataFrame(new_features, columns=feature_cols)

# Predict activity
y_pred_log = model.predict(X_new)
y_pred_ic50 = 10 ** y_pred_log

# Add predictions
X_new['predicted_log_ic50'] = y_pred_log
X_new['predicted_ic50_nM'] = y_pred_ic50

# Sort by predicted potency
X_new = X_new.sort_values('predicted_ic50_nM')

print(f"\n========== TOP 10 PREDICTED MOLECULES ==========")
print(X_new.head(10).to_string())

# Save results
X_new.to_csv("../new_molecules_predictions.csv", index=False)
print(f"\n✅ Saved to: ../new_molecules_predictions.csv")

# Filter highly potent molecules
highly_potent = X_new[X_new['predicted_ic50_nM'] < 10]
print(f"\n========== HIGHLY POTENT PREDICTIONS ==========")
print(f"Molecules with predicted IC50 < 10 nM: {len(highly_potent)}")

very_high = X_new[X_new['predicted_ic50_nM'] < 1]
print(f"Molecules with predicted IC50 < 1 nM: {len(very_high)}")

if len(very_high) > 0:
    print(f"\nTop 5 strongest predicted molecules:")
    print(very_high.head(5)[['predicted_ic50_nM']].to_string())

print("\n✅ Generation complete!")