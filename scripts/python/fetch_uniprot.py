"""
fetch_uniprot.py
Fetch protein information from UniProt API
"""

import requests
import pandas as pd
import time

class UniProtFetcher:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org"
    
    def get_protein_info(self, uniprot_id):
        """Get basic protein information"""
        url = f"{self.base_url}/uniprotkb/{uniprot_id}"
        response = requests.get(url, params={'format': 'json'})
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        info = {
            'uniprot_id': uniprot_id,
            'protein_name': None,
            'gene_name': None,
            'organism': None,
            'length': None,
            'function': None,
            'diseases': []
        }
        
        # Protein name
        if 'proteinDescription' in data:
            rec_name = data['proteinDescription'].get('recommendedName', {})
            info['protein_name'] = rec_name.get('fullName', {}).get('value')
        
        # Gene name
        if 'genes' in data and len(data['genes']) > 0:
            info['gene_name'] = data['genes'][0].get('geneName', {}).get('value')
        
        # Organism
        if 'organism' in data:
            info['organism'] = data['organism'].get('scientificName')
        
        # Sequence length
        if 'sequence' in data:
            info['length'] = data['sequence'].get('length')
        
        # Function and diseases
        for comment in data.get('comments', []):
            comment_type = comment.get('commentType')
            
            if comment_type == 'FUNCTION':
                texts = comment.get('texts', [])
                if texts:
                    info['function'] = texts[0].get('value')
            
            elif comment_type == 'DISEASE':
                diseases = comment.get('diseases', [])
                for disease in diseases:
                    disease_name = disease.get('diseaseId')
                    if disease_name:
                        info['diseases'].append(disease_name)
        
        return info
    
    def get_sequence(self, uniprot_id):
        """Get protein sequence in FASTA format"""
        url = f"{self.base_url}/uniprotkb/{uniprot_id}.fasta"
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.text
        return None
    
    def get_known_cancer_targets(self):
        """Get known cancer drug targets"""
        targets = [
            ('P00533', 'EGFR', 'Lung, breast, colorectal cancer'),
            ('P04637', 'TP53', 'Multiple cancers'),
            ('P04626', 'ERBB2', 'Breast cancer'),
            ('P35968', 'KDR', 'Angiogenesis in cancer'),
            ('P15692', 'VEGFA', 'Angiogenesis'),
            ('P07949', 'RET', 'Thyroid, lung cancer'),
            ('P08581', 'MET', 'Lung, gastric cancer'),
            ('P09619', 'PDGFRB', 'Leukemia, glioma'),
            ('P04049', 'RAF1', 'Melanoma, colorectal'),
            ('P11802', 'CDK4', 'Breast, melanoma'),
            ('P00533', 'EGFR', 'Lung, breast, colorectal cancer'),  # EGFR
            ('P04626', 'HER2', 'Breast cancer'),
            ('P35968', 'VEGFR2', 'Angiogenesis'),
            ('P07949', 'RET', 'Thyroid cancer'),
        ]
        
        results = []
        for uniprot_id, gene, cancers in targets:
            info = self.get_protein_info(uniprot_id)
            if info:
                results.append({
                    'uniprot_id': uniprot_id,
                    'protein_name': info['protein_name'],
                    'gene_name': gene,
                    'cancers': cancers,
                    'function': info['function'][:150] if info['function'] else ''
                })
            time.sleep(0.2)
        
        return pd.DataFrame(results)
    
    def get_interactions(self, uniprot_id):
        """Get protein-protein interactions"""
        url = f"{self.base_url}/uniprotkb/{uniprot_id}/interactions"
        response = requests.get(url, params={'format': 'json'})
        
        if response.status_code != 200:
            return pd.DataFrame()
        
        data = response.json()
        interactions = []
        
        for item in data.get('results', []):
            interactor = item.get('interactor', {})
            interactions.append({
                'uniprot_id': interactor.get('primaryAccession'),
                'protein_name': interactor.get('proteinName'),
                'gene_name': interactor.get('geneName'),
                'interaction_type': item.get('type')
            })
        
        return pd.DataFrame(interactions)

if __name__ == "__main__":
    print("=" * 60)
    print("UniProt API - Protein Information")
    print("=" * 60)
    
    fetcher = UniProtFetcher()
    
    # 1. Get info for EGFR
    print("\n1. EGFR Information:")
    print("-" * 40)
    egfr = fetcher.get_protein_info("P00533")
    if egfr:
        print(f"Protein: {egfr['protein_name']}")
        print(f"Gene: {egfr['gene_name']}")
        print(f"Organism: {egfr['organism']}")
        print(f"Length: {egfr['length']} amino acids")
        print(f"Function: {egfr['function'][:200]}..." if egfr['function'] else "Function: Not found")
        print(f"Diseases: {egfr['diseases'] if egfr['diseases'] else 'None found'}")
    
    # 2. Get known cancer targets
    print("\n2. Known Cancer Drug Targets:")
    print("-" * 40)
    cancer_targets = fetcher.get_known_cancer_targets()
    if not cancer_targets.empty:
        print(cancer_targets[['uniprot_id', 'gene_name', 'protein_name', 'cancers']].head(15).to_string(index=False))
    
    # 3. Get sequence for EGFR
    print("\n3. EGFR Sequence (first 300 characters):")
    print("-" * 40)
    sequence = fetcher.get_sequence("P00533")
    if sequence:
        lines = sequence.split('\n')
        print(lines[0])
        seq = ''.join(lines[1:])
        print(seq[:300] + "...")
    
    # 4. Save to file
    if not cancer_targets.empty:
        cancer_targets.to_csv("../../results/uniprot_cancer_targets.csv", index=False)
        print("\n✅ Cancer targets saved to: ../../results/uniprot_cancer_targets.csv")
    
    print("\n✅ UniProt API fetch complete!")