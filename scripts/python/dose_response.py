import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('ic50_cleaned.csv')

print("=== Dose-Response Analysis ===\n")

# Group by molecule and get best activity per molecule
best_per_molecule = df.loc[df.groupby('molecule_chembl_id')['ic50_nM'].idxmin()].copy()
best_per_molecule = best_per_molecule.sort_values('ic50_nM')

# Analyze top 10 most potent compounds
top10 = best_per_molecule.head(10)

print("Top 10 most potent compounds:")
print(top10[['molecule_chembl_id', 'ic50_nM']])
print()

# Distribution of pchembl values
valid_pchembl = df[df['pchembl_value'].notna()].copy()
print("pChEMBL Distribution:")
print(f"Min: {valid_pchembl['pchembl_value'].min()}")
print(f"Max: {valid_pchembl['pchembl_value'].max()}")
print(f"Mean: {valid_pchembl['pchembl_value'].mean():.2f}")
print(f"Median: {valid_pchembl['pchembl_value'].median():.2f}")
print()

# Identify outliers for further investigation
q1 = df['ic50_nM'].quantile(0.25)
q3 = df['ic50_nM'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df['ic50_nM'] < lower_bound) | (df['ic50_nM'] > upper_bound)]
print(f"Outliers detected: {len(outliers)} compounds")
print(f"Outlier range: > {upper_bound:.0f} nM or < {lower_bound:.0f} nM")
print()

# Statistical test: compare high vs low potency
high_potency = df[df['ic50_nM'] < 10]['pchembl_value'].dropna()
low_potency = df[df['ic50_nM'] > 1000]['pchembl_value'].dropna()

if len(high_potency) > 0 and len(low_potency) > 0:
    t_stat, p_value = stats.ttest_ind(high_potency, low_potency)
    print("=== Statistical Comparison (High vs Low Potency) ===")
    print(f"High potency mean pChEMBL: {high_potency.mean():.2f}")
    print(f"Low potency mean pChEMBL: {low_potency.mean():.2f}")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Significant difference between groups (p < 0.05)")
    else:
        print("Result: No significant difference")
print()

# Save top compounds for lead optimization
top_candidates = best_per_molecule.head(20)[['molecule_chembl_id', 'ic50_nM', 'pchembl_value']]
top_candidates.to_csv('lead_candidates.csv', index=False)
print("Saved top 20 lead candidates to lead_candidates.csv")