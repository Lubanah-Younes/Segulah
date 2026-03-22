import pandas as pd

# Load cleaned data
df = pd.read_csv('ic50_cleaned.csv')
leads = pd.read_csv('lead_candidates.csv')

# Add potency category if not exists
if 'potency_category' not in df.columns:
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

print("=== Preparing Data for Power BI Dashboard ===\n")

# Create summary table for dashboard
summary_table = pd.DataFrame({
    'metric': [
        'Total Compounds',
        'Unique Molecules',
        'High Potency Compounds (IC50 < 10 nM)',
        'High Potency Percentage',
        'Mean IC50 (nM)',
        'Median IC50 (nM)',
        'Best IC50 (nM)',
        'Best Molecule',
        'Lead Candidates Identified'
    ],
    'value': [
        len(df),
        df['molecule_chembl_id'].nunique(),
        len(df[df['ic50_nM'] < 10]),
        f"{len(df[df['ic50_nM'] < 10]) / len(df) * 100:.1f}%",
        f"{df['ic50_nM'].mean():.0f}",
        f"{df['ic50_nM'].median():.0f}",
        f"{df['ic50_nM'].min():.3f}",
        df.loc[df['ic50_nM'].idxmin(), 'molecule_chembl_id'],
        len(leads)
    ]
})

# Create potency distribution table
potency_dist = df['potency_category'].value_counts().reset_index()
potency_dist.columns = ['category', 'count']

# Create top compounds table for dashboard
top_compounds = df.nsmallest(20, 'ic50_nM')[
    ['molecule_chembl_id', 'ic50_nM', 'pchembl_value', 'standard_type', 'assay_chembl_id']
].copy()
top_compounds['ic50_nM'] = top_compounds['ic50_nM'].round(3)

# Save all tables for Power BI
summary_table.to_csv('dashboard_summary.csv', index=False)
potency_dist.to_csv('dashboard_potency_dist.csv', index=False)
top_compounds.to_csv('dashboard_top_compounds.csv', index=False)

print("Files created for Power BI:")
print("1. dashboard_summary.csv - Key metrics")
print("2. dashboard_potency_dist.csv - Potency distribution")
print("3. dashboard_top_compounds.csv - Top 20 compounds")

print("\n=== Preview of Summary Table ===")
print(summary_table.to_string(index=False))

print("\n=== Preview of Top 5 Compounds ===")
print(top_compounds.head().to_string(index=False))