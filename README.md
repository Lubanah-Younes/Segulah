# 🧬 SEGULAH AI - AI-Powered Drug Discovery Platform

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)](https://streamlit.io/)
[![RDKit](https://img.shields.io/badge/RDKit-2025.9-green)](https://rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🔬 Overview

**SEGULAH AI** is an advanced artificial intelligence platform designed to accelerate drug discovery by predicting the potency (IC50) of molecules and generating novel compounds. The platform is specifically tuned for **EGFR (Epidermal Growth Factor Receptor)** , a critical oncogenic target in **lung and breast cancer**.

The system uses a trained Neural Network (R² = 0.620) to evaluate molecular properties and predict biological activity, enabling researchers to screen thousands of molecules in seconds without costly laboratory experiments.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **🔮 Predict IC50** | Enter a SMILES string and get instant IC50 prediction |
| **🧪 Generate Molecules** | Create new molecules by adjusting molecular properties |
| **📊 Interactive Dashboard** | View statistics, potency distribution, and correlation charts |
| **🔍 Advanced Filters** | Filter molecules by IC50, ADMET score, potency class, and more |
| **🌙 Dark/Light Mode** | Toggle between themes for comfortable viewing |
| **📥 Export Results** | Download filtered molecules as CSV |

---

## 📊 Model Performance

| Model | Test R² | Training R² |
|-------|---------|-------------|
| **Neural Network** | **0.620** | 0.652 |
| Tuned Random Forest | 0.607 | 0.850 |
| XGBoost | 0.534 | 0.924 |

*Cross-validation (5-fold): 0.608 ± 0.032*

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Total molecules generated** | 1,000 |
| **Drug-like candidates** | 29 (Lipinski-compliant) |
| **Best known IC50** | 0.006 nM (CHEMBL53711) |
| **Best generated IC50** | 5.86 nM |
| **Worst generated IC50** | 660,950.8 nM |

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Machine Learning** | TensorFlow, Scikit-learn, XGBoost |
| **Cheminformatics** | RDKit (Molecular Descriptors) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Streamlit |
| **Cloud Storage** | Google BigQuery |
| **APIs** | ChEMBL, UniProt, PubChem, PDB |

---

## 📁 Project Structure
segulah-ai/
├── web_app/ # Streamlit application
│ ├── app.py # Main application
│ ├── requirements.txt # Python dependencies
│ ├── models/ # Trained models
│ ├── data/ # Generated molecules data
│ └── assets/ # Logo and images
├── scripts/ # Python and R scripts
│ ├── python/ # Data collection & ETL
│ └── r/ # Statistical analysis
├── generative_ai/ # AI model training
├── data/ # Processed datasets
├── results/ # Output files and plots
└── dashboard/ # Power BI dashboard

text

---

## 🧪 How It Works

### 1. Data Collection
- **479 compounds** with real IC50 values from ChEMBL
- Molecular features calculated using RDKit
- Target: EGFR (P00533) - lung and breast cancer marker

### 2. Model Training
- **Neural Network** with 3 hidden layers
- 8 molecular descriptors: MolWt, LogP, TPSA, H-Bond donors/acceptors, etc.
- Test R² = 0.620

### 3. Molecule Generation
- Generate **1,000 novel molecules** using the trained model
- Predict IC50 for each new molecule
- Evaluate drug-likeness using Lipinski's Rule of Five

### 4. ADMET Prediction
- Lipinski violations count
- ADMET score (0-100)
- Potency classification: Super Potent → Weak

---

## 🌐 Live Demo

Try the platform live: [**SEGULAH AI**](https://segulah-ai.streamlit.app)

---

## 📥 Installation (Local)

```bash
# Clone the repository
git clone https://github.com/your-username/segulah-ai.git
cd segulah-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r web_app/requirements.txt

# Run the app
streamlit run web_app/app.py
📚 External Data Sources
Source	Data Type
ChEMBL	Compound bioactivity (IC50, SMILES)
UniProt	Protein information, diseases, function
PubChem	Molecular properties, SMILES
PDB	3D protein structures
📊 Citation
If you use SEGULAH AI in your research, please cite:

text
SEGULAH AI: AI-Powered Drug Discovery Platform for EGFR Inhibition.
Available at: https://github.com/your-username/segulah-ai
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👩‍💻 Author
LUBANAH YOUNES 081227

Accelerating drug discovery through artificial intelligence.

🙏 Acknowledgments
ChEMBL for providing open-access bioactivity data

RDKit for cheminformatics tools

Streamlit for the web framework

TensorFlow for deep learning capabilities

📞 Contact
lubanahyounes@gmai.com