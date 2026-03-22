import pandas as pd

df = pd.read_csv('chembl_activities.csv')

print("=== Dataset Info ===")
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\n=== Data Types ===")
print(df.dtypes)
print("\n=== First 10 rows ===")
print(df.head(10))
print("\n=== Standard Types Distribution ===")
print(df['standard_type'].value_counts())
print("\n=== Summary Statistics ===")
print(df[['standard_value', 'pchembl_value']].describe())