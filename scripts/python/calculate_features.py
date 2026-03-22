"""
calculate_features.py
Calculate real molecular features from SMILES using RDKit
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import time

def calculate_features(smiles):
    """
    Calculate molecular descriptors from SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'TPSA': Descriptors.TPSA(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol)
    }
    
    return features

print("=" * 60)
print("Calculating Real Features from SMILES")
print("=" * 60)

# Load data with SMILES
df = pd.read_csv("../../data/processed/real_compounds_with_smiles.csv")
print(f"Loaded {len(df)} compounds with SMILES")

# Calculate features for each compound
features_list = []
valid_indices = []

print("\nCalculating molecular features...")
for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"  Processed {i}/{len(df)}")
    
    feats = calculate_features(row['smiles'])
    if feats:
        features_list.append(feats)
        valid_indices.append(i)

print(f"Valid compounds: {len(features_list)}")

# Create feature DataFrame
feature_df = pd.DataFrame(features_list)
feature_df['molecule_chembl_id'] = df.iloc[valid_indices]['molecule_chembl_id'].values
feature_df['ic50_nM'] = df.iloc[valid_indices]['ic50_nM'].values
feature_df['log_ic50'] = np.log10(feature_df['ic50_nM'])

# Save
output_path = "../../data/processed/real_data_with_real_features.csv"
feature_df.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")
print(f"   Shape: {feature_df.shape}")

print("\nFirst 5 rows:")
print(feature_df.head())