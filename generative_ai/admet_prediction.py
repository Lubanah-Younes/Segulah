"""
admet_prediction.py
Predict ADMET properties for generated molecules
ADMET = Absorption, Distribution, Metabolism, Excretion, Toxicity
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import os

def calculate_admet(smiles):
    """
    Calculate ADMET-related properties from SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {
        # Absorption & Distribution
        'MolWt': Descriptors.MolWt(mol),  # Molecular weight (<500 for good absorption)
        'LogP': Descriptors.MolLogP(mol),  # Lipophilicity (<5 for good absorption)
        'TPSA': Descriptors.TPSA(mol),  # Polar surface area (<140 for good absorption)
        'NumHDonors': Descriptors.NumHDonors(mol),  # H-bond donors (<5 for good absorption)
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # H-bond acceptors (<10 for good absorption)
        
        # Lipinski's Rule of Five
        'Lipinski_Violations': 0,
        
        # Toxicity indicators
        'PAINS_Alert': 0,  # Pan Assay Interference Compounds
        'BRENK_Alert': 0,  # Brenk alerts (toxicophores)
    }
    
    # Lipinski's Rule of Five violations
    violations = 0
    if properties['MolWt'] > 500: violations += 1
    if properties['LogP'] > 5: violations += 1
    if properties['NumHDonors'] > 5: violations += 1
    if properties['NumHAcceptors'] > 10: violations += 1
    properties['Lipinski_Violations'] = violations
    
    # Simple toxicity alerts (simulated)
    if properties['MolWt'] > 600: properties['PAINS_Alert'] += 1
    if properties['LogP'] > 5.5: properties['PAINS_Alert'] += 1
    if properties['NumAromaticRings'] > 5: properties['PAINS_Alert'] += 1
    
    return properties

print("=" * 60)
print("ADMET Prediction for Generated Molecules")
print("=" * 60)

# Load generated molecules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "results", "new_molecules_real_model.csv")
df = pd.read_csv(input_path)

print(f"Loaded {len(df)} generated molecules")

# Calculate ADMET properties (simulated features)
# In real project, you'd have actual SMILES
feature_cols = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                'NumAromaticRings', 'TPSA']

# Calculate ADMET rules
df['Lipinski_Violations'] = 0
df['Lipinski_Violations'] += (df['MolWt'] > 500).astype(int)
df['Lipinski_Violations'] += (df['LogP'] > 5).astype(int)
df['Lipinski_Violations'] += (df['NumHDonors'] > 5).astype(int)
df['Lipinski_Violations'] += (df['NumHAcceptors'] > 10).astype(int)

# Drug-likeness score (lower violations = better)
df['Drug_likeness'] = 5 - df['Lipinski_Violations']
df['Drug_likeness'] = df['Drug_likeness'].clip(0, 5)

# Toxicity score (simulated - lower is better)
df['Toxicity_Score'] = (df['MolWt'] / 600) + (df['LogP'] / 10) + (df['NumAromaticRings'] / 8)
df['Toxicity_Score'] = df['Toxicity_Score'].clip(0, 1)

# Overall score (higher is better)
df['Overall_Score'] = (df['Drug_likeness'] / 5) * 0.7 + (1 - df['Toxicity_Score']) * 0.3

print("\n========== ADMET SUMMARY ==========")
print(f"Molecules with 0 Lipinski violations: {len(df[df['Lipinski_Violations'] == 0])}")
print(f"Molecules with 1 Lipinski violation: {len(df[df['Lipinski_Violations'] == 1])}")
print(f"Molecules with 2+ Lipinski violations: {len(df[df['Lipinski_Violations'] >= 2])}")

# Best candidates
best_candidates = df.nsmallest(10, 'predicted_ic50_nM')
best_candidates = best_candidates.nlargest(10, 'Overall_Score')

print("\n========== TOP 10 CANDIDATES (Potency + Drug-likeness) ==========")
print(best_candidates[['predicted_ic50_nM', 'Lipinski_Violations', 
                        'Drug_likeness', 'Toxicity_Score', 'Overall_Score']].head(10))

# Save results
output_path = os.path.join(project_root, "results", "admet_results.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ ADMET results saved to: {output_path}")

# Highlight best candidates for lab testing
best_for_lab = best_candidates[best_candidates['predicted_ic50_nM'] < 20]
best_for_lab = best_for_lab[best_for_lab['Lipinski_Violations'] <= 1]

print(f"\n========== 🚀 RECOMMENDED FOR LAB TESTING ==========")
print(f"Candidates with IC50 < 20 nM and ≤1 Lipinski violation: {len(best_for_lab)}")

if len(best_for_lab) > 0:
    print("\nTop 5 recommended molecules:")
    print(best_for_lab[['predicted_ic50_nM', 'Lipinski_Violations', 
                         'Drug_likeness', 'Toxicity_Score']].head(5).to_string())

print("\n✅ ADMET analysis complete!")