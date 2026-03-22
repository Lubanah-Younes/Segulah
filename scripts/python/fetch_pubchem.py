"""
fetch_pubchem.py
Fetch compound information from PubChem API
"""

import requests
import pandas as pd
import time

class PubChemFetcher:
    def __init__(self):
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def get_smiles(self, cid):
        """Get canonical SMILES for a compound"""
        url = f"{self.base_url}/compound/cid/{cid}/property/CanonicalSMILES/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                props = data.get('PropertyTable', {}).get('Properties', [])
                if props:
                    return props[0].get('CanonicalSMILES')
            return None
        except Exception as e:
            print(f"Error fetching SMILES for CID {cid}: {e}")
            return None
    
    def get_compound_by_cid(self, cid):
        """Get compound information by CID"""
        url = f"{self.base_url}/compound/cid/{cid}/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            data = response.json()
            props = data.get('PC_Compounds', [{}])[0].get('props', [])
            
            # Extract properties
            info = {
                'cid': cid,
                'smiles': None,
                'molecular_formula': None,
                'molecular_weight': None,
                'logp': None,
                'h_bond_donor': None,
                'h_bond_acceptor': None,
                'rotatable_bonds': None,
                'tpsa': None
            }
            
            for prop in props:
                urn = prop.get('urn', {})
                label = urn.get('label')
                name = urn.get('name')
                
                value = prop.get('value', {})
                if 'ival' in value:
                    val = value['ival']
                elif 'fval' in value:
                    val = value['fval']
                elif 'sval' in value:
                    val = value['sval']
                else:
                    continue
                
                if label == 'Molecular Formula':
                    info['molecular_formula'] = val
                elif label == 'Molecular Weight':
                    info['molecular_weight'] = val
                elif label == 'SMILES' and name == 'Canonical':
                    info['smiles'] = val
                elif label == 'Log P':
                    info['logp'] = val
                elif label == 'Hydrogen Bond Donor Count':
                    info['h_bond_donor'] = val
                elif label == 'Hydrogen Bond Acceptor Count':
                    info['h_bond_acceptor'] = val
                elif label == 'Rotatable Bond Count':
                    info['rotatable_bonds'] = val
                elif label == 'Topological Polar Surface Area':
                    info['tpsa'] = val
            
            # Get SMILES separately if not found
            if not info['smiles']:
                info['smiles'] = self.get_smiles(cid)
            
            return info
        
        except Exception as e:
            print(f"Error fetching CID {cid}: {e}")
            return None
    
    def search_by_name(self, name):
        """Search compound by name"""
        url = f"{self.base_url}/compound/name/{name}/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            data = response.json()
            cid = data.get('PC_Compounds', [{}])[0].get('id', {}).get('id', {}).get('cid')
            
            if cid:
                return self.get_compound_by_cid(cid)
            return None
        
        except Exception as e:
            print(f"Error searching {name}: {e}")
            return None
    
    def search_by_smiles(self, smiles):
        """Search compound by SMILES"""
        url = f"{self.base_url}/compound/smiles/{smiles}/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            data = response.json()
            cid = data.get('PC_Compounds', [{}])[0].get('id', {}).get('id', {}).get('cid')
            
            if cid:
                return self.get_compound_by_cid(cid)
            return None
        
        except Exception as e:
            print(f"Error searching SMILES: {e}")
            return None
    
    def get_bioactivity(self, cid):
        """Get bioactivity data for compound"""
        url = f"{self.base_url}/compound/cid/{cid}/bioactivity/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            activities = []
            
            for act in data.get('BioActivities', []):
                activities.append({
                    'cid': cid,
                    'target_name': act.get('TargetName'),
                    'activity_value': act.get('ActivityValue'),
                    'activity_unit': act.get('ActivityUnit'),
                    'activity_type': act.get('ActivityType')
                })
            
            return pd.DataFrame(activities)
        
        except Exception as e:
            print(f"Error fetching bioactivity for CID {cid}: {e}")
            return pd.DataFrame()
    
    def get_properties_batch(self, cid_list):
        """Get properties for multiple compounds at once"""
        cids = ','.join([str(cid) for cid in cid_list])
        url = f"{self.base_url}/compound/cid/{cids}/property/CanonicalSMILES,MolecularFormula,MolecularWeight,LogP/JSON"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return pd.DataFrame()
            
            data = response.json()
            props = data.get('PropertyTable', {}).get('Properties', [])
            
            return pd.DataFrame(props)
        
        except Exception as e:
            print(f"Error fetching batch properties: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    print("=" * 60)
    print("PubChem API - Compound Information")
    print("=" * 60)
    
    fetcher = PubChemFetcher()
    
    # 1. Get info for a known compound
    print("\n1. Compound Information (Aspirin):")
    print("-" * 40)
    aspirin = fetcher.search_by_name("aspirin")
    if aspirin:
        print(f"CID: {aspirin['cid']}")
        print(f"SMILES: {aspirin['smiles']}")
        print(f"Formula: {aspirin['molecular_formula']}")
        print(f"Molecular Weight: {aspirin['molecular_weight']}")
        print(f"LogP: {aspirin['logp']}")
        print(f"H-Bond Donors: {aspirin['h_bond_donor']}")
        print(f"H-Bond Acceptors: {aspirin['h_bond_acceptor']}")
        print(f"Rotatable Bonds: {aspirin['rotatable_bonds']}")
        print(f"TPSA: {aspirin['tpsa']}")
    
    # 2. Get info for gefitinib (EGFR inhibitor)
    print("\n2. EGFR Inhibitor (Gefitinib):")
    print("-" * 40)
    gefitinib = fetcher.search_by_name("gefitinib")
    if gefitinib:
        print(f"CID: {gefitinib['cid']}")
        print(f"SMILES: {gefitinib['smiles']}")
        print(f"Formula: {gefitinib['molecular_formula']}")
        print(f"Molecular Weight: {gefitinib['molecular_weight']}")
        print(f"LogP: {gefitinib['logp']}")
    
    # 3. Get info for erlotinib (another EGFR inhibitor)
    print("\n3. EGFR Inhibitor (Erlotinib):")
    print("-" * 40)
    erlotinib = fetcher.search_by_name("erlotinib")
    if erlotinib:
        print(f"CID: {erlotinib['cid']}")
        print(f"SMILES: {erlotinib['smiles']}")
        print(f"Formula: {erlotinib['molecular_formula']}")
        print(f"Molecular Weight: {erlotinib['molecular_weight']}")
        print(f"LogP: {erlotinib['logp']}")
    
    # 4. Get batch properties for multiple compounds
    print("\n4. Batch Properties for EGFR Inhibitors:")
    print("-" * 40)
    cids = [2244, 123631, 176870]  # Aspirin, Gefitinib, Erlotinib
    batch_df = fetcher.get_properties_batch(cids)
    if not batch_df.empty:
        print(batch_df[['CID', 'CanonicalSMILES', 'MolecularFormula', 'MolecularWeight', 'LogP']].to_string(index=False))
    
    # 5. Save to files
    if gefitinib:
        gefitinib_df = pd.DataFrame([gefitinib])
        gefitinib_df.to_csv("../../results/pubchem_gefitinib.csv", index=False)
        print("\n✅ Gefitinib saved to: ../../results/pubchem_gefitinib.csv")
    
    if erlotinib:
        erlotinib_df = pd.DataFrame([erlotinib])
        erlotinib_df.to_csv("../../results/pubchem_erlotinib.csv", index=False)
        print("✅ Erlotinib saved to: ../../results/pubchem_erlotinib.csv")
    
    if not batch_df.empty:
        batch_df.to_csv("../../results/pubchem_batch_properties.csv", index=False)
        print("✅ Batch properties saved to: ../../results/pubchem_batch_properties.csv")
    
    print("\n✅ PubChem API fetch complete!")