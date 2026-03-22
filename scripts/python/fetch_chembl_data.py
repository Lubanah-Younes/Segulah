import requests
import pandas as pd
from datetime import datetime

class ChEMBLDataFetcher:
    
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/chembl/api/data"
        self.targets_endpoint = f"{self.base_url}/target"
        self.activities_endpoint = f"{self.base_url}/activity"
    
    def fetch_target_by_uniprot(self, uniprot_id):
        params = {
            'target_components__accession': uniprot_id,
            'format': 'json'
        }
        
        response = requests.get(self.targets_endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('targets', [])
        else:
            print(f"Error: {response.status_code}")
            return []
    
    def fetch_activities_for_target(self, target_chembl_id, limit=100):
        params = {
            'target_chembl_id': target_chembl_id,
            'format': 'json',
            'limit': limit
        }
        
        response = requests.get(self.activities_endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            activities = data.get('activities', [])
            
            df = pd.json_normalize(activities)
            
            relevant_columns = [
                'activity_id', 'assay_chembl_id', 'target_chembl_id',
                'molecule_chembl_id', 'standard_type', 'standard_value',
                'standard_units', 'standard_relation', 'pchembl_value'
            ]
            
            existing_cols = [col for col in relevant_columns if col in df.columns]
            df = df[existing_cols]
            
            df['extraction_date'] = datetime.now()
            
            return df
        else:
            print(f"Error fetching activities: {response.status_code}")
            return pd.DataFrame()


if __name__ == "__main__":
    fetcher = ChEMBLDataFetcher()
    
    uniprot_id = "P00533"  # EGFR
    targets = fetcher.fetch_target_by_uniprot(uniprot_id)
    
    if targets:
        target_chembl_id = targets[0]['target_chembl_id']
        print(f"Target found: {target_chembl_id}")
        
        activities_df = fetcher.fetch_activities_for_target(target_chembl_id, limit=500)
        print(f"Fetched {len(activities_df)} activities")
        print(activities_df.head())
        
        activities_df.to_csv('chembl_activities.csv', index=False)
        print("Data saved to chembl_activities.csv")