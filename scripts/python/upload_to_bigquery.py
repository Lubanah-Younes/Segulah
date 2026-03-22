import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os

print("=== Uploading Data to BigQuery ===\n")

# Load the data
df_ic50 = pd.read_csv('ic50_cleaned.csv')
df_summary = pd.read_csv('dashboard_summary.csv')
df_potency = pd.read_csv('dashboard_potency_dist.csv')
df_top = pd.read_csv('dashboard_top_compounds.csv')

print(f"Loaded {len(df_ic50)} IC50 records")
print(f"Loaded {len(df_summary)} summary records")
print(f"Loaded {len(df_potency)} potency records")
print(f"Loaded {len(df_top)} top compounds")

# For now, save locally as Parquet (can upload to BigQuery later)
df_ic50.to_parquet('ic50_cleaned.parquet', index=False)
df_summary.to_parquet('dashboard_summary.parquet', index=False)
df_potency.to_parquet('dashboard_potency_dist.parquet', index=False)
df_top.to_parquet('dashboard_top_compounds.parquet', index=False)

print("\nFiles saved as Parquet format:")
print("1. ic50_cleaned.parquet")
print("2. dashboard_summary.parquet")
print("3. dashboard_potency_dist.parquet")
print("4. dashboard_top_compounds.parquet")

# Note: To upload to BigQuery, you need:
# 1. Google Cloud account
# 2. Service account key file
# 3. Enable BigQuery API

print("\n=== Next Steps for BigQuery ===")
print("1. Create Google Cloud account (free tier available)")
print("2. Create a project")
print("3. Enable BigQuery API")
print("4. Create service account and download JSON key")
print("5. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
print("6. Run the upload code above")

print("\nData ready for dashboard visualization")