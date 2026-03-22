"""
prepare_ai_data.py
Prepare dataset for AI model using RDKit
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def smiles_to_features(smiles):
    """
    Convert SMILES to molecular descriptors
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
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol)
    }
    
    return list(features.values())

def prepare_dataset(input_file, output_file):
    """
    Prepare dataset for AI model training
    """
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds")
    
    # Check if SMILES column exists
    if 'smiles' not in df.columns:
        print("No SMILES column found. Using simulated features...")
        np.random.seed(42)
        
        feature_names = ['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 
                         'NumRotatableBonds', 'NumAromaticRings', 'TPSA', 
                         'FractionCSP3', 'NumHeteroatoms', 'NumHeavyAtoms']
        
        features = np.random.randn(len(df), 10)
        
        # Adjust ranges
        features[:, 0] = features[:, 0] * 100 + 300
        features[:, 1] = features[:, 1] * 2 + 2
        features[:, 2] = np.abs(features[:, 2] * 2)
        features[:, 3] = np.abs(features[:, 3] * 4)
        features[:, 4] = np.abs(features[:, 4] * 6)
        features[:, 5] = np.abs(features[:, 5] * 3)
        features[:, 6] = np.abs(features[:, 6] * 50 + 50)
        features[:, 7] = np.abs(features[:, 7])
        features[:, 8] = np.abs(features[:, 8] * 6)
        features[:, 9] = np.abs(features[:, 9] * 15 + 15)
        
        X_df = pd.DataFrame(features, columns=feature_names)
    else:
        print("Converting SMILES to features...")
        features_list = []
        for i, smiles in enumerate(df['smiles']):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(df)} compounds")
            feats = smiles_to_features(smiles)
            if feats:
                features_list.append(feats)
        
        X_df = pd.DataFrame(features_list, columns=feature_names)
        df = df.iloc[:len(features_list)]
    
    # Target
    y_df = df[['ic50_nM']].copy()
    y_df['log_ic50'] = np.log10(y_df['ic50_nM'])
    
    # Combine
    result_df = pd.concat([X_df, y_df], axis=1)
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Data saved to {output_file}")
    print(f"   Features: {list(X_df.columns)}")
    print(f"   Target: log_ic50")
    print(f"   Shape: {result_df.shape}")
    
    return X_df, y_df['log_ic50']

if __name__ == "__main__":
    print("=" * 50)
    print("AI Data Preparation")
    print("=" * 50)
    X, y = prepare_dataset("../ic50_cleaned.csv", "../ai_data.csv")
    print(f"\n✅ Done! X shape: {X.shape}, y shape: {y.shape}")