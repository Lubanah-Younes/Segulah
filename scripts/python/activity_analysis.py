import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('ic50_cleaned.csv')

print("=== Activity Distribution ===")

# Categorize compounds by potency
def categorize_potency(ic50_nM):
    if ic50_nM < 10:
        return 'High (IC50 < 10 nM)'
    elif ic50_nM < 100:
        return 'Medium-High (10-100 nM)'
    elif ic50_nM < 1000:
        return 'Medium (100-1000 nM)'
    elif ic50_nM < 10000:
        return 'Low-Medium (1-10 uM)'
    else:
        return 'Low (>10 uM)'

df['potency_category'] = df['ic50_nM'].apply(categorize_potency)

print("\nPotency distribution:")
print(df['potency_category'].value_counts())
print(f"\nPercentage of high potency compounds (<10 nM):")
high_potency_pct = (df['potency_category'] == 'High (IC50 < 10 nM)').mean() * 100
print(f"{high_potency_pct:.1f}%")

# Group by molecule to find best compound per molecule
best_per_molecule = df.loc[df.groupby('molecule_chembl_id')['ic50_nM'].idxmin()]
best_per_molecule = best_per_molecule.sort_values('ic50_nM')

print(f"\n=== Top 5 Molecules (by best activity) ===")
print(best_per_molecule[['molecule_chembl_id', 'ic50_nM', 'pchembl_value']].head())

# Save analysis results
summary = {
    'total_compounds': len(df),
    'unique_molecules': df['molecule_chembl_id'].nunique(),
    'high_potency_percentage': high_potency_pct,
    'mean_ic50_nM': df['ic50_nM'].mean(),
    'median_ic50_nM': df['ic50_nM'].median(),
    'best_ic50_nM': df['ic50_nM'].min(),
    'best_molecule': df.loc[df['ic50_nM'].idxmin(), 'molecule_chembl_id']
}

print("\n=== Summary Statistics ===")
for key, value in summary.items():
    print(f"{key}: {value}")

# Save summary to file
pd.DataFrame([summary]).to_csv('activity_summary.csv', index=False)
print("\nSaved summary to activity_summary.csv")