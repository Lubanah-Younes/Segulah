"""
SEGULAH AI - Advanced AI-Powered Drug Discovery Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors
import base64
from PIL import Image
import numpy as np
from io import BytesIO
import time
import hashlib

# Page config
st.set_page_config(
    page_title="SEGULAH AI - Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'notifications_shown' not in st.session_state:
    st.session_state.notifications_shown = False

def add_notification(msg, type="info"):
    st.session_state.notifications.append({"msg": msg, "type": type, "time": time.time()})
    if len(st.session_state.notifications) > 5:
        st.session_state.notifications.pop(0)

def show_notifications():
    for notif in st.session_state.notifications[-5:]:
        if notif["type"] == "success":
            st.success(notif["msg"])
        elif notif["type"] == "error":
            st.error(notif["msg"])
        elif notif["type"] == "warning":
            st.warning(notif["msg"])
        else:
            st.info(notif["msg"])

# ============================================
# CUSTOM CSS - Modern Dark/Light Theme
# ============================================
if st.session_state.dark_mode:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background: #0a0c10; }
        .modern-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 32px;
            padding: 48px 40px;
            margin-bottom: 32px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .modern-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
        }
        .modern-subtitle {
            font-size: 1.1rem;
            color: #94a3b8;
            font-weight: 400;
        }
        .glass-card-modern {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 24px;
            padding: 24px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.3s ease;
        }
        .glass-card-modern:hover {
            border-color: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }
        .metric-modern {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.2s ease;
        }
        .metric-modern:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        .metric-value-modern {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        .metric-label-modern {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
            margin-top: 8px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #0a0c10 100%);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        [data-testid="stSidebar"] * { color: #e2e8f0; }
        [data-testid="stSidebar"] .stSelectbox div {
            color: white !important;
            font-weight: 500;
        }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 40px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(59,130,246,0.3);
        }
        .dataframe {
            background: rgba(30, 41, 59, 0.5) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }
        h1, h2, h3 {
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .footer-modern {
            background: linear-gradient(135deg, #0f172a 0%, #0a0c10 100%);
            border-radius: 24px;
            padding: 24px;
            margin-top: 48px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .footer-modern p { color: #64748b; margin: 0; }
        .footer-modern small { color: #475569; }
    </style>
    """
else:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background: #f9fafb; }
        .modern-hero {
            background: linear-gradient(135deg, #f1f5f9 0%, #e9eef3 100%);
            border-radius: 32px;
            padding: 48px 40px;
            margin-bottom: 32px;
        }
        .modern-title {
            font-size: 4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
        }
        .modern-subtitle {
            font-size: 1.1rem;
            color: #475569;
            font-weight: 400;
        }
        .glass-card-modern {
            background: #ffffff;
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.03);
            border: 1px solid #eef2f6;
            transition: all 0.3s ease;
        }
        .glass-card-modern:hover {
            box-shadow: 0 8px 20px rgba(0,0,0,0.05);
            transform: translateY(-2px);
        }
        .metric-modern {
            background: #ffffff;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            border: 1px solid #eef2f6;
            transition: all 0.2s ease;
        }
        .metric-modern:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        .metric-value-modern {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }
        .metric-label-modern {
            font-size: 0.8rem;
            color: #64748b;
            font-weight: 500;
            margin-top: 8px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #eef2f6;
        }
        [data-testid="stSidebar"] * { color: #0f172a; }
        [data-testid="stSidebar"] .stSelectbox div {
            color: #0f172a !important;
            font-weight: 500;
        }
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            border: none;
            border-radius: 40px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(59,130,246,0.3);
        }
        .dataframe {
            border-radius: 16px !important;
            border: 1px solid #eef2f6 !important;
        }
        h1, h2, h3 { font-weight: 700; color: #0f172a; }
        .footer-modern {
            background: #ffffff;
            border-radius: 24px;
            padding: 24px;
            margin-top: 48px;
            text-align: center;
            border: 1px solid #eef2f6;
        }
        .footer-modern p { color: #64748b; margin: 0; }
        .footer-modern small { color: #94a3b8; }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# ============================================
# LOGO FUNCTIONS
# ============================================
def remove_white_background(img, threshold=240):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    data = np.array(img)
    white_pixels = (data[:, :, 0] > threshold) & \
                   (data[:, :, 1] > threshold) & \
                   (data[:, :, 2] > threshold)
    data[white_pixels, 3] = 0
    return Image.fromarray(data)

def get_logo_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "assets", "segulah_logo.png")
    if os.path.exists(logo_path):
        return logo_path
    return None

LOGO_PATH = get_logo_path()

def get_logo_transparent(width=80):
    if not LOGO_PATH:
        return None
    try:
        img = Image.open(LOGO_PATH)
        img = remove_white_background(img)
        return img
    except:
        return None

# ============================================
# LOAD DATA & MODELS
# ============================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "neural_network_model.keras")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except:
        return None, None

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "smiles_generated.csv")
    return pd.read_csv(data_path)

with st.spinner("Loading SEGULAH AI..."):
    time.sleep(0.5)
    model, scaler = load_model()
    df = load_data()
    add_notification("SEGULAH AI ready!", "success")

# Add ADMET Score to dataframe
def calculate_admet_score(row):
    score = 0
    if row['MolWt'] < 500: score += 25
    if row['LogP'] < 5: score += 25
    if row['NumHDonors'] < 5: score += 25
    if row['NumHAcceptors'] < 10: score += 25
    return score

df['ADMET_Score'] = df.apply(calculate_admet_score, axis=1)
df['Potency_Class'] = pd.cut(df['predicted_ic50_nM'], 
                               bins=[0, 1, 10, 100, 1000, float('inf')],
                               labels=['Super Potent', 'Highly Potent', 'Potent', 'Moderate', 'Weak'])

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    # Theme Toggle - single button
    if st.button("🌙 Toggle Dark Mode", key="theme_btn", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    # Logo
    logo_img = get_logo_transparent(50)
    if logo_img:
        st.image(logo_img, width=50)
    else:
        st.markdown("🧬")
    st.markdown("### SEGULAH AI")
    st.markdown("*Intelligent Drug Discovery*")
    st.markdown("---")
    
    option = st.selectbox(
        "🔬 Navigation",
        ["🏠 Dashboard", "🔮 Predict IC50", "🧪 Generate Molecules", "📊 View Molecules", "📈 About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Platform Stats")
    st.metric("Total Molecules", f"{len(df):,}")
    st.metric("Best IC50", f"{df['predicted_ic50_nM'].min():.3f} nM")
    st.metric("Drug-like", f"{len(df[df['ADMET_Score'] >= 75])} molecules")

# ============================================
# HERO SECTION WITH LOGO
# ============================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    logo_img = get_logo_transparent(180)
    if logo_img:
        st.image(logo_img, width=180)
    else:
        st.markdown("🧬")
with col_title:
    st.markdown("""
    <div class="modern-hero" style="margin-left: 0;">
        <div class="modern-title">SEGULAH AI</div>
        <div class="modern-subtitle">AI-Powered Drug Discovery Platform | Targeting EGFR</div>
    </div>
    """, unsafe_allow_html=True)

# Show notifications only once at startup
if not st.session_state.notifications_shown:
    show_notifications()
    st.session_state.notifications_shown = True

# ============================================
# 1. DASHBOARD
# ============================================
if option == "🏠 Dashboard":
    st.markdown("## Dashboard")
    st.markdown("---")
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="metric-value-modern">{len(df):,}</div>
            <div class="metric-label-modern">Total Molecules Generated</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="metric-value-modern">{df['predicted_ic50_nM'].min():.3f}</div>
            <div class="metric-label-modern">Best IC50 (nM)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="metric-value-modern">{df['ADMET_Score'].mean():.0f}</div>
            <div class="metric-label-modern">Avg ADMET Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-modern">
            <div class="metric-value-modern">0.620</div>
            <div class="metric-label-modern">Model R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card-modern">', unsafe_allow_html=True)
        st.subheader("🎯 Potency Distribution")
        
        potency_counts = df['Potency_Class'].value_counts().reset_index()
        potency_counts.columns = ['Class', 'Count']
        
        fig = px.pie(potency_counts, values='Count', names='Class',
                     color_discrete_sequence=px.colors.sequential.Blues_r,
                     title="Distribution by Potency")
        fig.update_layout(showlegend=True, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card-modern">', unsafe_allow_html=True)
        st.subheader("📈 IC50 Distribution")
        
        fig = px.histogram(df, x='predicted_ic50_nM', nbins=50,
                          title="Histogram of Predicted IC50 Values",
                          color_discrete_sequence=['#3b82f6'])
        fig.update_layout(showlegend=False, height=400, 
                         xaxis_title="IC50 (nM)", yaxis_title="Frequency",
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Correlation Chart
    st.markdown('<div class="glass-card-modern">', unsafe_allow_html=True)
    st.subheader("🔬 Molecular Properties Correlation")
    
    corr_data = df[['MolWt', 'LogP', 'NumAromaticRings', 'predicted_ic50_nM', 'ADMET_Score']].corr()
    fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                    color_continuous_scale='Blues',
                    title="Feature Correlation Matrix")
    fig.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Scatter Plot
    st.markdown('<div class="glass-card-modern">', unsafe_allow_html=True)
    st.subheader("📊 Molecular Weight vs IC50")
    
    fig_scatter = px.scatter(df, 
                              x='MolWt', 
                              y='predicted_ic50_nM',
                              color='Potency_Class',
                              size='ADMET_Score',
                              hover_data=['LogP', 'NumAromaticRings'],
                              title="Relationship between Molecular Weight and IC50",
                              color_discrete_sequence=px.colors.sequential.Blues_r,
                              labels={'MolWt': 'Molecular Weight (g/mol)', 
                                     'predicted_ic50_nM': 'IC50 (nM)'})
    
    fig_scatter.update_layout(height=500,
                              xaxis_title="Molecular Weight (g/mol)",
                              yaxis_title="IC50 (nM)",
                              legend_title="Potency Class",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top Molecules
    st.markdown('<div class="glass-card-modern">', unsafe_allow_html=True)
    st.subheader("🏆 Top 10 Most Potent Molecules")
    
    top10 = df.nsmallest(10, 'predicted_ic50_nM')[['predicted_ic50_nM', 'MolWt', 'LogP', 'NumAromaticRings', 'ADMET_Score']].round(3)
    st.dataframe(top10, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# 2. PREDICT IC50
# ============================================
elif option == "🔮 Predict IC50":
    st.markdown("## 🔮 Predict IC50 from SMILES")
    st.markdown("---")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check model files.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            smiles = st.text_area("Enter SMILES string:", 
                                  "c1ccc2ccccc2c1COCOCOCS(=O)C",
                                  height=100)
            
            if st.button("🔮 Predict IC50", type="primary", use_container_width=True):
                with st.spinner("Analyzing molecule..."):
                    time.sleep(0.5)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        features = [
                            Descriptors.MolWt(mol),
                            Descriptors.MolLogP(mol),
                            Descriptors.NumHDonors(mol),
                            Descriptors.NumHAcceptors(mol),
                            Descriptors.NumRotatableBonds(mol),
                            Descriptors.NumAromaticRings(mol),
                            Descriptors.TPSA(mol),
                            Descriptors.FractionCSP3(mol)
                        ]
                        
                        features_scaled = scaler.transform([features])
                        pred_log = model.predict(features_scaled)[0][0]
                        pred_ic50 = 10 ** pred_log
                        
                        st.success(f"### 🎯 Predicted IC50: **{pred_ic50:.3f} nM**")
                        add_notification(f"Predicted IC50: {pred_ic50:.3f} nM", "success")
                        
                        violations = sum([features[0] > 500, features[1] > 5, 
                                         features[2] > 5, features[3] > 10])
                        admet_score = 100 - (violations * 25)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if violations == 0:
                                st.success("✅ Lipinski: Passed (0 violations)")
                            else:
                                st.warning(f"⚠️ Lipinski: {violations} violations")
                        
                        with col_b:
                            st.metric("ADMET Score", f"{admet_score}/100")
                    else:
                        st.error("❌ Invalid SMILES!")
                        add_notification("Invalid SMILES entered", "error")
        
        with col2:
            st.markdown("### 💡 Examples")
            st.code("c1ccc2ccccc2c1COCOCOCS(=O)C", language="python")
            st.caption("Molecule #0 - 5.86 nM")
            st.code("COc1cc2c(cc1OCCCN3CCOCC3)ncnc2Nc4ccc(F)c(Cl)c4", language="python")
            st.caption("Gefitinib - ~10 nM")

# ============================================
# 3. GENERATE MOLECULES
# ============================================
elif option == "🧪 Generate Molecules":
    st.markdown("## 🧪 Generate Molecules from Properties")
    st.markdown("---")
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check model files.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            molwt = st.slider("Molecular Weight (MolWt)", 200, 600, 400, help="200-600 is optimal")
            logp = st.slider("LogP", -2.0, 5.0, 2.5, help="2-3 is ideal for drugs")
            h_donors = st.slider("H-Bond Donors", 0, 5, 2, help="<5 for good absorption")
            h_acceptors = st.slider("H-Bond Acceptors", 0, 10, 5, help="<10 for drug-likeness")
        
        with col2:
            rot_bonds = st.slider("Rotatable Bonds", 0, 12, 5, help="<10 for better bioavailability")
            aromatic_rings = st.slider("Aromatic Rings", 0, 4, 2, help="1-3 rings typical")
            tpsa = st.slider("TPSA", 20, 140, 80, help="<140 for good absorption")
            fraction_csp3 = st.slider("Fraction CSP3", 0.0, 1.0, 0.5, help="Higher = more 3D")
        
        if st.button("🎯 Predict IC50", type="primary", use_container_width=True):
            with st.spinner("Predicting..."):
                time.sleep(0.5)
                features = [[molwt, logp, h_donors, h_acceptors, 
                             rot_bonds, aromatic_rings, tpsa, fraction_csp3]]
                
                features_scaled = scaler.transform(features)
                pred_log = model.predict(features_scaled)[0][0]
                pred_ic50 = 10 ** pred_log
                
                st.success(f"### 🎯 Predicted IC50: **{pred_ic50:.3f} nM**")
                add_notification(f"Generated molecule with IC50: {pred_ic50:.3f} nM", "success")
                
                violations = sum([molwt > 500, logp > 5, h_donors > 5, h_acceptors > 10])
                admet_score = 100 - (violations * 25)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if violations == 0:
                        st.success("✅ Lipinski: Passed")
                    else:
                        st.warning(f"⚠️ {violations} violations")
                with col_b:
                    if pred_ic50 < 10:
                        st.success("✅ Potency: High")
                    elif pred_ic50 < 100:
                        st.info("📊 Potency: Medium")
                    else:
                        st.warning("⚠️ Potency: Low")
                with col_c:
                    st.metric("ADMET Score", f"{admet_score}/100")

# ============================================
# 4. VIEW MOLECULES
# ============================================
elif option == "📊 View Molecules":
    st.markdown("## 📊 Generated Molecules")
    st.markdown("---")
    
    # Advanced Filters
    with st.expander("🔍 Advanced Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_ic50 = st.number_input("Min IC50 (nM)", 0.0, 1000.0, 0.0, step=1.0)
            max_ic50 = st.number_input("Max IC50 (nM)", 0.0, 1000.0, 50.0, step=1.0)
        
        with col2:
            min_molwt = st.number_input("Min MolWt", 200, 600, 200)
            max_molwt = st.number_input("Max MolWt", 200, 600, 600)
            min_logp = st.number_input("Min LogP", -2.0, 5.0, -2.0, step=0.5)
            max_logp = st.number_input("Max LogP", -2.0, 5.0, 5.0, step=0.5)
        
        with col3:
            min_admet = st.slider("Min ADMET Score", 0, 100, 50)
            potency_filter = st.multiselect("Potency Class", 
                                            ['Super Potent', 'Highly Potent', 'Potent', 'Moderate', 'Weak'],
                                            default=['Super Potent', 'Highly Potent'])
    
    # Apply filters
    filtered = df[
        (df['predicted_ic50_nM'] >= min_ic50) & 
        (df['predicted_ic50_nM'] <= max_ic50) &
        (df['MolWt'] >= min_molwt) &
        (df['MolWt'] <= max_molwt) &
        (df['LogP'] >= min_logp) &
        (df['LogP'] <= max_logp) &
        (df['ADMET_Score'] >= min_admet) &
        (df['Potency_Class'].isin(potency_filter))
    ]
    
    st.markdown(f"### 📈 Results: **{len(filtered)}** molecules found")
    
    # Sort options
    sort_by = st.selectbox("Sort by:", ["predicted_ic50_nM", "ADMET_Score", "MolWt", "LogP"])
    filtered = filtered.sort_values(sort_by)
    
    # Display
    display_cols = ['predicted_ic50_nM', 'ADMET_Score', 'Potency_Class', 'MolWt', 'LogP', 'NumAromaticRings']
    st.dataframe(filtered[display_cols].round(3), use_container_width=True)
    
    # Download
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="segulah_molecules.csv",
        mime="text/csv",
        use_container_width=True
    )

# ============================================
# 5. ABOUT
# ============================================
elif option == "📈 About":
    st.markdown("## 📈 About SEGULAH AI")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🧬 SEGULAH AI - Next-Generation Drug Discovery
        
        **SEGULAH AI** is an advanced artificial intelligence platform designed to accelerate the discovery of novel therapeutics targeting the **Epidermal Growth Factor Receptor (EGFR)** , a critical oncogenic driver in **lung and breast cancer**.
        
        ---
        
        ### 🔬 Technical Architecture
        
        | Component | Technology |
        |-----------|------------|
        | **ML Framework** | TensorFlow 2.x |
        | **Molecular Features** | RDKit (Molecular Descriptors) |
        | **Data Storage** | Google BigQuery |
        | **Visualization** | Plotly + Streamlit |
        | **Model Type** | Deep Neural Network |
        
        ---
        
        ### 📊 Model Performance Metrics
        
        | Metric | Value |
        |--------|-------|
        | Test R² | **0.620** |
        | Training R² | 0.652 |
        | Cross-Validation R² | 0.608 ± 0.032 |
        | Best Known IC50 | 0.006 nM (CHEMBL53711) |
        | Best Generated IC50 | 5.86 nM |
        
        ---
        
        ### 🎯 Key Achievements
        
        - ✅ **1,000** novel molecules generated
        - ✅ **29** drug-like candidates identified (Lipinski-compliant)
        - ✅ **Neural Network** outperformed Random Forest (R² +0.046)
        - ✅ **Integrated** 4 external APIs (ChEMBL, UniProt, PubChem, PDB)
        - ✅ **ADMET prediction** with Lipinski's Rule of Five
        """)
    
    with col2:
        st.markdown("""
        <div class="glass-card-modern">
            <h3>🛠️ Technologies</h3>
            <ul>
                <li>Python 3.12</li>
                <li>TensorFlow</li>
                <li>RDKit</li>
                <li>Scikit-learn</li>
                <li>Pandas/NumPy</li>
                <li>Plotly</li>
                <li>Streamlit</li>
            </ul>
            <hr>
            <h3>📊 Data Sources</h3>
            <ul>
                <li>ChEMBL (479 compounds)</li>
                <li>UniProt (Protein data)</li>
                <li>PubChem (Molecular info)</li>
                <li>PDB (3D structures)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
footer_logo_html = ""
if LOGO_PATH:
    try:
        img = Image.open(LOGO_PATH)
        img_transparent = remove_white_background(img)
        buf = BytesIO()
        img_transparent.save(buf, format='PNG')
        logo_base64 = base64.b64encode(buf.getvalue()).decode()
        footer_logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="30" style="border-radius: 8px;">'
    except:
        footer_logo_html = "🧬"
else:
    footer_logo_html = "🧬"

st.markdown(f"""
<div class="footer-modern">
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; margin-bottom: 8px;">
        {footer_logo_html}
        <p><strong>SEGULAH AI</strong> | AI-Powered Drug Discovery Platform</p>
    </div>
    <small>Target: EGFR | Neural Network Model R² = 0.620 | Best Generated IC50 = 5.86 nM</small>
</div>
""", unsafe_allow_html=True)