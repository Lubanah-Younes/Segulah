"""
fetch_pdb.py
Fetch protein structure information from PDB API
"""

import requests
import pandas as pd
import time

class PDBApi:
    def __init__(self):
        self.base_url = "https://data.rcsb.org/rest/v1"
        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    def get_structure_info(self, pdb_id):
        """Get detailed information about a PDB structure"""
        url = f"{self.base_url}/core/entry/{pdb_id}"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            info = {
                'pdb_id': pdb_id,
                'title': data.get('struct', {}).get('title'),
                'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined'),
                'method': data.get('rcsb_entry_info', {}).get('experimental_method'),
                'deposition_date': data.get('rcsb_accession_info', {}).get('deposit_date'),
                'polymer_count': len(data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])),
                'ligand_count': len(data.get('rcsb_entry_container_identifiers', {}).get('nonpolymer_entity_ids', []))
            }
            
            return info
        
        except Exception as e:
            print(f"Error fetching {pdb_id}: {e}")
            return None
    
    def get_ligands(self, pdb_id):
        """Get ligands bound to the structure"""
        url = f"{self.base_url}/core/entry/{pdb_id}"
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return []
            
            data = response.json()
            ligands = []
            
            nonpolymer_ids = data.get('rcsb_entry_container_identifiers', {}).get('nonpolymer_entity_ids', [])
            
            for lig_id in nonpolymer_ids:
                lig_data = data.get('nonpolymer_entities', {}).get(lig_id, {})
                chem_comp = lig_data.get('rcsb_chem_comp_descriptor', {})
                
                ligands.append({
                    'ligand_id': lig_id,
                    'name': chem_comp.get('chem_comp_name'),
                    'formula': lig_data.get('entity_nonpolymer', {}).get('formula'),
                    'smiles': chem_comp.get('smiles_string'),
                    'inchi': chem_comp.get('inchi_string')
                })
            
            return ligands
        
        except Exception as e:
            print(f"Error fetching ligands for {pdb_id}: {e}")
            return []
    
    def download_pdb_file(self, pdb_id, save_path):
        """Download PDB file for structure"""
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'w') as f:
                    f.write(response.text)
                return True
            return False
        
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("PDB API - EGFR Structure Information")
    print("=" * 60)
    
    pdb = PDBApi()
    
    # Known EGFR structures with inhibitors
    egfr_structures = {
        '1M17': 'Gefitinib (ZD1839)',
        '2ITN': 'Erlotinib (Tarceva)',
        '3POZ': 'Afatinib',
        '4HJO': 'Osimertinib (AZD9291)',
        '5HG5': 'Lapatinib',
        '5UGC': 'Neratinib',
        '6DUK': 'Dacomitinib',
        '4WKQ': 'CO-1686 (Rociletinib)'
    }
    
    results = []
    
    print("\n1. EGFR Structures with Known Inhibitors:")
    print("-" * 80)
    print(f"{'PDB ID':<8} {'Inhibitor':<20} {'Resolution':<12} {'Method':<15} {'Ligands'}")
    print("-" * 80)
    
    for pdb_id, inhibitor in egfr_structures.items():
        info = pdb.get_structure_info(pdb_id)
        if info:
            results.append({
                'pdb_id': pdb_id,
                'inhibitor': inhibitor,
                'resolution': info['resolution'],
                'method': info['method'],
                'ligand_count': info['ligand_count']
            })
            print(f"{pdb_id:<8} {inhibitor:<20} {str(info['resolution']):<12} {str(info['method'])[:15]:<15} {info['ligand_count']}")
        else:
            print(f"{pdb_id:<8} {inhibitor:<20} {'Not found':<12}")
        time.sleep(0.2)
    
    print("\n2. Ligands Bound in Each Structure:")
    print("-" * 80)
    
    for pdb_id, inhibitor in egfr_structures.items():
        ligands = pdb.get_ligands(pdb_id)
        if ligands:
            print(f"\n{pdb_id} ({inhibitor}):")
            for lig in ligands:
                print(f"  - {lig['ligand_id']}: {lig['name']}")
                if lig['smiles']:
                    print(f"    SMILES: {lig['smiles'][:80]}...")
    
    # Download one structure
    print("\n3. Downloading PDB File:")
    print("-" * 40)
    pdb_id = '1M17'
    save_path = f"../../results/{pdb_id}.pdb"
    if pdb.download_pdb_file(pdb_id, save_path):
        print(f"✅ Downloaded {pdb_id}.pdb to {save_path}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv("../../results/pdb_egfr_structures.csv", index=False)
        print(f"\n✅ Saved to: ../../results/pdb_egfr_structures.csv")
    
    print("\n✅ PDB API fetch complete!")