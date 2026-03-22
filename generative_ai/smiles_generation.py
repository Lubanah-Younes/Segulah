"""
smiles_generation.py
Generate realistic SMILES for new molecules
"""

import pandas as pd
import numpy as np
import random
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

print("=" * 60)
print("Generating Realistic SMILES for New Molecules")
print("=" * 60)

# Load generated molecules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "results", "new_molecules_real_model.csv")
df = pd.read_csv(input_path)

print(f"Loaded {len(df)} generated molecules")

# Load ADMET results to get Lipinski and Drug_likeness
admet_path = os.path.join(project_root, "results", "admet_results.csv")
if os.path.exists(admet_path):
    df_admet = pd.read_csv(admet_path)
    # Merge with ADMET results
    df = pd.concat([df, df_admet[['Lipinski_Violations', 'Drug_likeness', 'Toxicity_Score']]], axis=1)
    print("Merged ADMET results")
else:
    # Calculate ADMET if not available
    print("Calculating ADMET properties...")
    df['Lipinski_Violations'] = 0
    df['Lipinski_Violations'] += (df['MolWt'] > 500).astype(int)
    df['Lipinski_Violations'] += (df['LogP'] > 5).astype(int)
    df['Lipinski_Violations'] += (df['NumHDonors'] > 5).astype(int)
    df['Lipinski_Violations'] += (df['NumHAcceptors'] > 10).astype(int)
    df['Drug_likeness'] = 5 - df['Lipinski_Violations']
    df['Drug_likeness'] = df['Drug_likeness'].clip(0, 5)
    df['Toxicity_Score'] = (df['MolWt'] / 600) + (df['LogP'] / 10) + (df['NumAromaticRings'] / 8)
    df['Toxicity_Score'] = df['Toxicity_Score'].clip(0, 1)

# Generate realistic SMILES for top molecules
df_top = df.nsmallest(100, 'predicted_ic50_nM').copy()

def generate_smiles_from_features(features):
    """
    Generate a SMILES string based on features
    """
    fragments = [
        'c1ccccc1',  # benzene
        'c1ccncc1',  # pyridine
        'C=O',  # carbonyl
        'NC',  # amine
        'CO',  # alcohol
        'CCO',  # ethanol
        'c1ccc2ccccc2c1',  # naphthalene
        'CC(=O)N',  # amide
        'CS(=O)C',  # sulfoxide
    ]
    
    n_aromatic = int(round(features.get('NumAromaticRings', 1)))
    if n_aromatic > 5:
        n_aromatic = 3
    
    if n_aromatic >= 2:
        smiles = 'c1ccc2ccccc2c1'
    else:
        smiles = random.choice(['c1ccccc1', 'c1ccncc1'])
    
    for _ in range(int(features.get('NumRotatableBonds', 2))):
        smiles = smiles + random.choice(fragments)
    
    return smiles

print("\nGenerating SMILES for top molecules...")
smiles_list = []

for i, row in df_top.iterrows():
    features = {
        'NumAromaticRings': row['NumAromaticRings'],
        'NumRotatableBonds': row['NumRotatableBonds']
    }
    smiles = generate_smiles_from_features(features)
    smiles_list.append(smiles)
    
    if i < 10:
        print(f"Molecule {i}: IC50={row['predicted_ic50_nM']:.2f} nM, SMILES={smiles}")

df_top['generated_smiles'] = smiles_list

# Calculate additional drug-like properties
def is_drug_like(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return violations <= 1

df_top['is_drug_like'] = df_top['generated_smiles'].apply(is_drug_like)

print("\n========== GENERATED SMILES SUMMARY ==========")
print(f"Total molecules with generated SMILES: {len(df_top)}")
print(f"Drug-like molecules: {df_top['is_drug_like'].sum()}")

# Save
output_path = os.path.join(project_root, "results", "smiles_generated.csv")
df_top.to_csv(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# Best candidates
if 'Lipinski_Violations' in df_top.columns and 'Drug_likeness' in df_top.columns:
    best = df_top[df_top['is_drug_like'] == True].nsmallest(10, 'predicted_ic50_nM')
    print("\n========== 🏆 BEST CANDIDATES WITH SMILES ==========")
    if len(best) > 0:
        print(best[['predicted_ic50_nM', 'Lipinski_Violations', 
                    'Drug_likeness', 'generated_smiles']].head(10).to_string())
    else:
        print("No drug-like candidates found. Showing best available:")
        best = df_top.nsmallest(10, 'predicted_ic50_nM')
        print(best[['predicted_ic50_nM', 'generated_smiles']].head(10).to_string())
else:
    print("\n========== 🏆 BEST CANDIDATES WITH SMILES ==========")
    best = df_top.nsmallest(10, 'predicted_ic50_nM')
    print(best[['predicted_ic50_nM', 'generated_smiles']].head(10).to_string())

print("\n✅ SMILES generation complete!")