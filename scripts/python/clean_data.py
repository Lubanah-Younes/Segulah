import pandas as pd
import numpy as np

df = pd.read_csv('chembl_activities.csv')

print("=== Before cleaning ===")
print(f"Total records: {len(df)}")

# Filter only IC50 data
df_ic50 = df[df['standard_type'] == 'IC50'].copy()

print(f"\n=== After filtering IC50 only ===")
print(f"IC50 records: {len(df_ic50)}")

# Convert IC50 values to nM (nanomolar) for consistency
def convert_to_nm(value, units):
    if pd.isna(value) or pd.isna(units):
        return np.nan
    if units == 'nM':
        return value
    elif units == 'uM':
        return value * 1000
    elif units == 'pM':
        return value / 1000
    elif units == 'M':
        return value * 1e9
    else:
        return np.nan

df_ic50['ic50_nM'] = df_ic50.apply(
    lambda x: convert_to_nm(x['standard_value'], x['standard_units']), 
    axis=1
)

# Remove invalid values
df_ic50_clean = df_ic50[df_ic50['ic50_nM'].notna()].copy()

# Log transform for better visualization
df_ic50_clean['log_ic50_nM'] = np.log10(df_ic50_clean['ic50_nM'])

print(f"\n=== After cleaning ===")
print(f"Valid IC50 records: {len(df_ic50_clean)}")
print(f"\nIC50 summary (nM):")
print(df_ic50_clean['ic50_nM'].describe())
print(f"\nLog IC50 summary:")
print(df_ic50_clean['log_ic50_nM'].describe())

# Save cleaned data
df_ic50_clean.to_csv('ic50_cleaned.csv', index=False)
print("\nSaved to ic50_cleaned.csv")

# Show top 10 most potent compounds
top_potent = df_ic50_clean.nsmallest(10, 'ic50_nM')[['molecule_chembl_id', 'ic50_nM', 'pchembl_value']]
print("\n=== Top 10 most potent compounds ===")
print(top_potent)