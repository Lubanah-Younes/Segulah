# Data Lineage - Drug Discovery Platform

## 1. Source Data (ChEMBL)
- **API:** ChEMBL (https://www.ebi.ac.uk/chembl/)
- **Target:** EGFR (CHEMBL203)
- **Script:** `scripts/python/fetch_real_data.py`
- **Output:** `data/processed/real_compounds_with_smiles.csv`
- **Records:** 479 compounds with SMILES and IC50

## 2. Feature Engineering
- **Script:** `scripts/python/calculate_features.py`
- **Input:** `data/processed/real_compounds_with_smiles.csv`
- **Output:** `data/processed/real_data_with_real_features.csv`
- **Features:** MolWt, LogP, NumHDonors, NumHAcceptors, NumRotatableBonds, NumAromaticRings, TPSA, FractionCSP3

## 3. Model Training

### 3.1 Random Forest
- **Script:** `generative_ai/train_real_features.py`
- **Input:** `data/processed/real_data_with_real_features.csv`
- **Output:** `results/real_features_model.pkl`
- **Performance:** Test R² = 0.574

### 3.2 Hyperparameter Tuning
- **Script:** `generative_ai/hyperparameter_tuning.py`
- **Input:** `data/processed/real_data_with_real_features.csv`
- **Output:** `results/tuned_rf_model.pkl`
- **Performance:** Test R² = 0.607

### 3.3 XGBoost
- **Script:** `generative_ai/xgboost_model.py`
- **Input:** `data/processed/real_data_with_real_features.csv`
- **Output:** `results/xgboost_model.pkl`
- **Performance:** Test R² = 0.534

### 3.4 Neural Network
- **Script:** `generative_ai/neural_network.py`
- **Input:** `data/processed/real_data_with_real_features.csv`
- **Output:** `results/neural_network_model.keras`
- **Performance:** Test R² = 0.620 (Best)

## 4. Cross Validation
- **Script:** `generative_ai/cross_validation.py`
- **Input:** `data/processed/real_data_with_real_features.csv`
- **Output:** `results/cv_results.csv`, `results/cross_validation_results.png`
- **Performance:** CV R² = 0.608 ± 0.032

## 5. ADMET Prediction
- **Script:** `generative_ai/admet_prediction.py`
- **Input:** `results/new_molecules_real_model.csv`
- **Output:** `results/admet_results.csv`
- **Rules:** Lipinski's Rule of Five

## 6. Molecule Generation

### 6.1 Feature-Based Generation
- **Script:** `generative_ai/generate_new_molecules.py`
- **Input:** `results/real_features_model.pkl`
- **Output:** `results/new_molecules_real_model.csv`
- **Molecules:** 1000 new molecules with predicted IC50

### 6.2 SMILES Generation
- **Script:** `generative_ai/smiles_generation.py`
- **Input:** `results/new_molecules_real_model.csv`
- **Output:** `results/smiles_generated.csv`
- **SMILES:** Realistic chemical structures

## 7. External Data Sources

### 7.1 UniProt
- **Script:** `scripts/python/fetch_uniprot.py`
- **Output:** `results/uniprot_cancer_targets.csv`
- **Information:** Protein function, diseases, gene names

### 7.2 PubChem
- **Script:** `scripts/python/fetch_pubchem.py`
- **Output:** `results/pubchem_gefitinib.csv`, `results/pubchem_erlotinib.csv`, `results/pubchem_batch_properties.csv`
- **Information:** Molecular formula, weight, LogP

### 7.3 PDB
- **Script:** `scripts/python/fetch_pdb.py`
- **Output:** `results/1M17.pdb`, `results/pdb_egfr_structures.csv`
- **Information:** 3D structures of EGFR with inhibitors

## 8. Statistical Analysis (R)

### 8.1 T-Test
- **Script:** `scripts/r/r_analysis.R`
- **Input:** `data/processed/ic50_cleaned.csv`
- **Output:** `results/r_results.txt`
- **Result:** p-value < 2.2e-16

### 8.2 DESeq2
- **Script:** `scripts/r/deseq2_analysis.R`
- **Output:** `results/deseq2_results.csv`, `results/deseq2_plot.png`

### 8.3 Proteomics
- **Script:** `scripts/r/proteomics_simple.R`
- **Output:** `results/proteomics_results.csv`, `results/proteomics_volcano.png`

## 9. Visualization

### 9.1 Power BI Dashboard
- **File:** `Drug Discovery Platform - EGFR.pbix`
- **Data Sources:** 
  - `data/processed/ic50_cleaned.csv`
  - `data/processed/top_compounds.csv`
  - `data/processed/dashboard_summary.csv`
  - `data/processed/dashboard_potency_dist.csv`

### 9.2 Python Plots
- **Location:** `results/`
- **Files:** `neural_network_results.png`, `cross_validation_results.png`, `model_comparison.png`

---

## Data Flow Diagram
┌─────────────────────────────────────────────────────────────┐
│ │
▼ │
ChEMBL API ──► real_compounds_with_smiles.csv ──► calculate_features.py ──► real_data_with_real_features.csv
│ │
│ │
├─────────────────────────────────────────────────────────────┤
│ │
▼ ▼
Neural Network ──► neural_network_model.keras Random Forest ──► real_features_model.pkl
│ │
│ │
▼ ▼
generate_new_molecules.py ◄─────────────────────────────────────────────────────┘
│
▼
new_molecules_real_model.csv
│
▼
smiles_generation.py
│
▼
smiles_generated.csv

External Data:

UniProt API ──► uniprot_cancer_targets.csv
PubChem API ──► pubchem_gefitinib.csv
PDB API ──────► 1M17.pdb, pdb_egfr_structures.csv

Statistical Analysis (R):

ic50_cleaned.csv ──► r_analysis.R ──► r_results.txt
──► deseq2_analysis.R ──► deseq2_results.csv
──► proteomics_simple.R ──► proteomics_results.csv

Visualization:

Data files ──► Power BI ──► Drug Discovery Platform - EGFR.pbix
Data files ──► Python plots ──► results/*.png

text

---

## Summary Table

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| Data Collection | ChEMBL API | fetch_real_data.py | real_compounds_with_smiles.csv |
| Feature Engineering | real_compounds_with_smiles.csv | calculate_features.py | real_data_with_real_features.csv |
| Model Training | real_data_with_real_features.csv | neural_network.py | neural_network_model.keras |
| Molecule Generation | neural_network_model.keras | generate_new_molecules.py | new_molecules_real_model.csv |
| SMILES Generation | new_molecules_real_model.csv | smiles_generation.py | smiles_generated.csv |
| ADMET | new_molecules_real_model.csv | admet_prediction.py | admet_results.csv |
| External Data | UniProt/PubChem/PDB APIs | fetch_*.py | Various CSV files |
| Statistics | ic50_cleaned.csv | R scripts | r_results.txt |
| Dashboard | CSV files | Power BI | .pbix file |