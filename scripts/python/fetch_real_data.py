"""
fetch_real_data.py
Fetch real compounds with SMILES from ChEMBL for EGFR target
"""

import requests
import pandas as pd
import time
import numpy as np

def fetch_compounds_with_smiles(target_chembl_id, limit=500):
    """
    Fetch compounds with IC50 and SMILES for a target
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    
    # Step 1: Get activities for target
    activities_url = f"{base_url}/activity"
    params = {
        'target_chembl_id': target_chembl_id,
        'format': 'json',
        'limit': limit,
        'standard_type': 'IC50'
    }
    
    print(f"Fetching activities for {target_chembl_id}...")
    response = requests.get(activities_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()
    
    data = response.json()
    activities = data.get('activities', [])
    print(f"Found {len(activities)} activities")
    
    # Step 2: Get SMILES for each compound
    compounds_data = []
    
    for i, act in enumerate(activities):
        if i % 50 == 0:
            print(f"Processing {i}/{len(activities)}...")
        
        mol_chembl_id = act.get('molecule_chembl_id')
        if not mol_chembl_id:
            continue
        
        # Fetch molecule details (includes SMILES)
        mol_url = f"{base_url}/molecule/{mol_chembl_id}"
        mol_response = requests.get(mol_url, params={'format': 'json'})
        
        if mol_response.status_code == 200:
            mol_data = mol_response.json()
            
            # Extract SMILES
            smiles = None
            if 'molecule_structures' in mol_data:
                structures = mol_data['molecule_structures']
                smiles = structures.get('canonical_smiles')
            
            # Get IC50 value and convert to float
            ic50_value = act.get('standard_value')
            if ic50_value:
                try:
                    ic50_value = float(ic50_value)
                except (ValueError, TypeError):
                    ic50_value = None
            
            compounds_data.append({
                'molecule_chembl_id': mol_chembl_id,
                'ic50_nM': ic50_value,
                'pchembl_value': act.get('pchembl_value'),
                'smiles': smiles,
                'assay_chembl_id': act.get('assay_chembl_id')
            })
        
        time.sleep(0.1)  # Be nice to the API
    
    # Convert to DataFrame
    df = pd.DataFrame(compounds_data)
    
    # Clean data
    df = df.dropna(subset=['ic50_nM', 'smiles'])
    df = df[df['ic50_nM'] > 0]
    
    print(f"\n✅ Fetched {len(df)} compounds with SMILES")
    return df

if __name__ == "__main__":
    # EGFR target ID
    target = "CHEMBL203"
    
    df = fetch_compounds_with_smiles(target, limit=500)
    
    if len(df) > 0:
        # Save to file
        import os
        output_path = "../../data/processed/real_compounds_with_smiles.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved to: {output_path}")
        print("\nFirst 5 compounds:")
        print(df[['molecule_chembl_id', 'ic50_nM', 'smiles']].head())