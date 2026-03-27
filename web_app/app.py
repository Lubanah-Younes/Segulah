"""
SEGULAH AI - Advanced AI-Powered Drug Discovery Platform
COMPLETE VERSION - All Features + Molecular Docking Viewer
Developer: LUBANAH H. YOUNES
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, rdDepictor, rdMolDescriptors
import base64
from PIL import Image
from io import BytesIO
import time
import hashlib
import json
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Force English language
st.markdown("""
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="google" content="notranslate">
        <meta http-equiv="Content-Language" content="en">
        <meta name="language" content="en">
    </head>
    <style>
        * {
            direction: ltr !important;
            text-align: left !important;
            unicode-bidi: normal !important;
        }
        .stApp, .stMarkdown, .stSelectbox, .stSidebar {
            direction: ltr !important;
        }
    </style>
""", unsafe_allow_html=True)

# Try to import py3Dmol for docking viewer
try:
    import py3Dmol
    import streamlit.components.v1 as components
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False
    components = None

# ============================================
# PAGE CONFIG - MUST BE FIRST
# ============================================
st.set_page_config(
    page_title="SEGULAH AI - Drug Discovery Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# FAVICON FUNCTIONS
# ============================================
def get_favicon_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    possible_paths = [
        os.path.join(base_dir, "assets", "segulah_logo.png"),
        os.path.join(base_dir, "assets", "favicon.png"),
        os.path.join(base_dir, "assets", "favicon.ico"),
        os.path.join(parent_dir, "assets", "segulah_logo.png"),
        os.path.join(parent_dir, "results", "lubatinib.png"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def set_custom_favicon():
    favicon_path = get_favicon_path()
    if favicon_path:
        try:
            with open(favicon_path, "rb") as f:
                img_data = f.read()
            if favicon_path.endswith('.png'):
                mime_type = "image/png"
            else:
                mime_type = "image/x-icon"
            favicon_base64 = base64.b64encode(img_data).decode()
            favicon_html = f"""
            <link rel="icon" type="{mime_type}" href="data:{mime_type};base64,{favicon_base64}">
            <link rel="shortcut icon" type="{mime_type}" href="data:{mime_type};base64,{favicon_base64}">
            """
            st.markdown(favicon_html, unsafe_allow_html=True)
        except:
            pass

set_custom_favicon()

# ============================================
# SESSION STATE
# ============================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'reading_mode' not in st.session_state:
    st.session_state.reading_mode = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'notifications_shown' not in st.session_state:
    st.session_state.notifications_shown = False
if 'current_option' not in st.session_state:
    st.session_state.current_option = " Dashboard"
if 'comparison_compounds' not in st.session_state:
    st.session_state.comparison_compounds = []
if 'reports_generated' not in st.session_state:
    st.session_state.reports_generated = 0
if 'push_notifications_enabled' not in st.session_state:
    st.session_state.push_notifications_enabled = False
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# ============================================
# PUSH NOTIFICATIONS FUNCTIONS
# ============================================
def request_push_permission():
    push_html = """
    <script>
        if ('Notification' in window) {
            Notification.requestPermission().then(function(permission) {
                if (permission === 'granted') {
                    console.log('Push notifications enabled');
                }
            });
        }
    </script>
    """
    st.components.v1.html(push_html, height=0)

def send_push_notification(title, body, icon="⚛️"):
    if st.session_state.push_notifications_enabled:
        push_html = f"""
        <script>
            if ('Notification' in window && Notification.permission === 'granted') {{
                new Notification("{title}", {{
                    body: "{body}",
                    icon: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E⚛️%3C/text%3E%3C/svg%3E"
                }});
            }}
        </script>
        """
        st.components.v1.html(push_html, height=0)

def enable_push_notifications():
    st.session_state.push_notifications_enabled = True
    request_push_permission()
    send_push_notification("SEGULAH AI", "Notifications enabled! You'll receive updates.", "🔔")

# ============================================
# PROGRESS BAR FUNCTIONS
# ============================================
def show_operation_progress(operation_name, duration=2):
    progress_bar = st.progress(0, text=f"🔄 {operation_name}...")
    for i in range(101):
        time.sleep(duration / 100)
        progress_bar.progress(i / 100, text=f"🔄 {operation_name}... {i}%")
    progress_bar.empty()
    return True

# ============================================
# NOTIFICATION FUNCTIONS
# ============================================
def add_notification(msg, type="info", push=False):
    st.session_state.notifications.append({"msg": msg, "type": type, "time": time.time()})
    if len(st.session_state.notifications) > 5:
        st.session_state.notifications.pop(0)
    
    if push and st.session_state.push_notifications_enabled:
        send_push_notification("SEGULAH AI", msg, "🔔" if type == "info" else "⚠️" if type == "warning" else "✅")

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

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def toggle_reading_mode():
    st.session_state.reading_mode = not st.session_state.reading_mode
    if st.session_state.reading_mode:
        add_notification("Reading mode activated - larger text, reduced distractions", "info")
    else:
        add_notification("Normal mode restored", "info")

# ============================================
# CELEBRATION FUNCTIONS
# ============================================
def show_confetti():
    confetti_html = """
    <script>
        if (typeof confetti === 'undefined') {
            var script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1';
            document.head.appendChild(script);
            script.onload = function() {
                canvasConfetti({ particleCount: 150, spread: 80, origin: { y: 0.6 }, colors: ['#3b82f6', '#10b981', '#f59e0b'] });
                setTimeout(function() { canvasConfetti({ particleCount: 100, spread: 100, origin: { y: 0.7 }, colors: ['#3b82f6', '#8b5cf6'] }); }, 250);
            }
        } else {
            canvasConfetti({ particleCount: 150, spread: 80, origin: { y: 0.6 }, colors: ['#3b82f6', '#10b981', '#f59e0b'] });
            setTimeout(function() { canvasConfetti({ particleCount: 100, spread: 100, origin: { y: 0.7 }, colors: ['#3b82f6', '#8b5cf6'] }); }, 250);
        }
    </script>
    """
    st.components.v1.html(confetti_html, height=0)
    if st.session_state.push_notifications_enabled:
        send_push_notification("🎉 Excellent Result!", "Congratulations on your discovery!", "🎉")

def show_balloons_celebration():
    try:
        st.balloons()
    except:
        pass

def show_toast_message(message, icon="🎉"):
    try:
        st.toast(message, icon=icon)
    except:
        st.info(message)

# ============================================
# LOADING ANIMATION
# ============================================
def show_loading_animation():
    loading_html = """
    <style>
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
        .loading-container { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 50px; }
        .loading-spinner { width: 60px; height: 60px; border: 4px solid rgba(59,130,246,0.2); border-top-color: #3b82f6; border-right-color: #8b5cf6; border-radius: 50%; animation: spin 0.8s linear infinite; }
        .loading-text { margin-top: 20px; font-size: 1rem; font-weight: 500; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: pulse 1.5s ease-in-out infinite; }
    </style>
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Loading SEGULAH AI</div>
    </div>
    """
    return loading_html

# ============================================
# LOGO FUNCTIONS
# ============================================
def remove_white_background(img, threshold=240):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    data = np.array(img)
    white_pixels = (data[:, :, 0] > threshold) & (data[:, :, 1] > threshold) & (data[:, :, 2] > threshold)
    near_white = (data[:, :, 0] > 240) & (data[:, :, 1] > 240) & (data[:, :, 2] > 240)
    white_pixels = white_pixels | near_white
    data[white_pixels, 3] = 0
    return Image.fromarray(data)

def get_logo_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    possible_paths = [
        os.path.join(base_dir, "assets", "segulah_logo.png"),
        os.path.join(base_dir, "assets", "logo.png"),
        os.path.join(parent_dir, "assets", "segulah_logo.png"),
        os.path.join(parent_dir, "results", "lubatinib.png"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def get_logo_transparent(width=150):
    logo_path = get_logo_path()
    if logo_path:
        try:
            img = Image.open(logo_path)
            img = remove_white_background(img)
            return img
        except:
            pass
    return None

def get_logo_html_transparent(width=35):
    logo_img = get_logo_transparent(width)
    if logo_img:
        try:
            buf = BytesIO()
            logo_img.save(buf, format='PNG')
            logo_base64 = base64.b64encode(buf.getvalue()).decode()
            return f'<img src="data:image/png;base64,{logo_base64}" width="{width}" style="border-radius: 6px;">'
        except:
            pass
    return "⚛️"

# ============================================
# 3D COORDINATES FUNCTION
# ============================================
def generate_3d_coordinates(mol):
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_h)
        mol_3d = Chem.RemoveHs(mol_h)
        return mol_3d
    except:
        return mol

# ============================================
# ADVANCED PROPERTIES FUNCTIONS
# ============================================
def calculate_quantum_properties(mol):
    if mol is None:
        return {}
    logp = Descriptors.MolLogP(mol)
    return {
        'HOMO_Estimate (eV)': -6.5 - 0.2 * logp,
        'LUMO_Estimate (eV)': -1.5 + 0.2 * logp,
        'Band_Gap (eV)': 5.0 - 0.4 * abs(logp),
        'Dipole_Moment (D)': 2.0 + 0.5 * logp
    }

def calculate_synthetic_accessibility(mol):
    if mol is None:
        return 10
    n_rings = Descriptors.NumAromaticRings(mol)
    n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    score = 5 + (n_rings * 0.3) + (n_stereo * 0.5) + (max(0, n_heavy - 30) * 0.1)
    return min(10, max(1, score))

def calculate_bbb_permeability(mol):
    if mol is None:
        return "Unknown"
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    if logp < 1:
        return "Low"
    elif logp > 3 and tpsa < 70:
        return "High"
    elif 1 <= logp <= 3 and tpsa < 90:
        return "Medium"
    else:
        return "Low"

# ============================================
# DOCKING VIEWER FUNCTIONS
# ============================================
def get_available_pdb_files():
    """Get available PDB files from results folder"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    pdb_files = []
    
    # Check results folder
    results_dir = os.path.join(parent_dir, "results")
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(results_dir, file))
    
    # Check web_app folder
    web_app_dir = base_dir
    for file in os.listdir(web_app_dir):
        if file.endswith('.pdb'):
            pdb_files.append(os.path.join(web_app_dir, file))
    
    # Check assets folder
    assets_dir = os.path.join(base_dir, "assets")
    if os.path.exists(assets_dir):
        for file in os.listdir(assets_dir):
            if file.endswith('.pdb'):
                pdb_files.append(os.path.join(assets_dir, file))
    
    return pdb_files

def predict_pk_parameters(mol):
    if mol is None:
        return {}
    logp = Descriptors.MolLogP(mol)
    return {
        'Half_Life (h)': max(0.5, min(24, 2.5 + 0.5 * logp)),
        'Clearance (L/h)': max(0.2, min(10, 0.8 + 0.3 * logp)),
        'Vd (L/kg)': max(0.2, min(5, 0.5 + 0.2 * logp)),
        'Bioavailability (%)': max(10, min(95, 70 - 5 * abs(logp - 2.5)))
    }

def predict_toxicity(mol):
    if mol is None:
        return {}
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return {
        'hERG Risk': 'High' if logp > 4.5 else 'Medium' if logp > 3 else 'Low',
        'Hepatotoxicity': 'High' if mw > 450 else 'Medium' if mw > 350 else 'Low',
        'Genotoxicity': 'High' if logp > 4 and mw > 400 else 'Low',
        'Carcinogenicity': 'Medium' if logp > 3.5 else 'Low'
    }

def get_mutation_resistance(smiles, mutations=['T790M', 'L858R', 'C797S']):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else smiles
    if mol is None:
        return {}
    logp = Descriptors.MolLogP(mol)
    resistance = {}
    for mut in mutations:
        if mut == 'T790M':
            score = max(0, min(100, 40 + 15 * (logp - 2)))
        elif mut == 'L858R':
            score = max(0, min(100, 35 + 12 * (logp - 2)))
        elif mut == 'C797S':
            score = max(0, min(100, 60 + 10 * (logp - 2)))
        else:
            score = 50
        resistance[mut] = score
    return resistance

# ============================================
# API FUNCTIONS
# ============================================
@st.cache_data(ttl=3600)
def fetch_chembl_data(compound_name):
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={compound_name}&format=json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('molecules', [])[:5]
    except:
        pass
    return []

@st.cache_data(ttl=3600)
def fetch_pubchem_data(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/MolecularWeight,LogP,HBondDonorCount,HBondAcceptorCount/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

# ============================================
# REPORT GENERATION
# ============================================
def generate_html_report(compound_name, smiles, properties, docking_results, pk_params, toxicity, resistance):
    current_year = datetime.now().year
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SEGULAH AI Report - {compound_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #2563eb; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
            .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚛️ SEGULAH AI Drug Discovery Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Compound:</strong> {compound_name}</p>
            <p><strong>SMILES:</strong> <code>{smiles}</code></p>
    """
    
    if properties:
        html += "<h2>Molecular Properties</h2>"
        html += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
        for key, val in properties.items():
            html += f"业<nth>{key}</th>\\n<td>{val}</td>\\n</tr>"
        html += "</table>"
    
    html += f"""
            <div class="footer">
                <p>Powered by LUBANAH H. YOUNES | Target: EGFR | Model R² = 0.620</p>
                <p>© {current_year} SEGULAH AI</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# ============================================
# FOOTER
# ============================================
def get_footer():
    current_year = datetime.now().year
    if st.session_state.dark_mode:
        text_color = "#94a3b8"
        accent = "#3b82f6"
    else:
        text_color = "#475569"
        accent = "#3b82f6"
    
    return f"""
    <div class="footer">
        <div>
            <span style="font-weight: 600; color: {accent};">LUBANAH H. YOUNES</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {text_color};">💻 Computational Drug Discovery</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {text_color};">🤖 AI Researcher</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {text_color};">🔬 Pharmaceutical Innovation</span>
        </div>
        <div style="margin-top: 2px;">
            <span style="color: {text_color};">© {current_year} SEGULAH AI</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {text_color};">🎯 Target: EGFR</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {text_color};">📊 Model R² = 0.620</span>
            <span style="color: {text_color};">|</span>
            <span style="color: {accent};">⭐ Lubatinib™</span>
        </div>
    </div>
    """

# ============================================
# CSS - COMPLETE WITH ENHANCED DARK MODE FIX
# ============================================
if st.session_state.dark_mode:
    if st.session_state.reading_mode:
        theme_css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Georgia', 'Times New Roman', serif; }
            .stApp { background: #1a1a2e; }
            p, li, .stMarkdown { font-size: 1.1rem !important; line-height: 1.8 !important; color: #e2e8f0 !important; }
            h1 { font-size: 2.5rem !important; color: white !important; }
            h2 { font-size: 2rem !important; color: white !important; }
            h3 { font-size: 1.5rem !important; color: white !important; }
            .card, .metric-card { background: #1e293b !important; border: 1px solid #334155 !important; }
            .hero { background: #0f172a; border: 1px solid #3b82f6; }
        </style>
        """
    else:
        theme_css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }
            .stApp { background: #0a0c10; }
            
            /* ========== FIX ALL WHITE BOXES ========== */
            .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
                background: #1e293b !important;
                border-left: 4px solid #3b82f6 !important;
            }
            .stAlert > div, .stAlert p {
                color: #e2e8f0 !important;
            }
            
            /* Fix metric cards */
            .metric-card {
                background: #1e293b !important;
                border: 1px solid #334155 !important;
                border-radius: 16px !important;
                padding: 20px !important;
            }
            .metric-value {
                color: #f1f5f9 !important;
                font-size: 2rem !important;
                font-weight: 700 !important;
            }
            .metric-label {
                color: #94a3b8 !important;
                font-size: 0.75rem !important;
            }
            
            /* Fix regular cards */
            .card {
                background: #1e293b !important;
                border: 1px solid #334155 !important;
                border-radius: 20px !important;
                padding: 24px !important;
            }
            
            /* Fix expander */
            .streamlit-expanderHeader {
                background: #1e293b !important;
                color: #e2e8f0 !important;
                border: 1px solid #334155 !important;
                border-radius: 12px !important;
            }
            .streamlit-expanderContent {
                background: #0f172a !important;
                border-radius: 12px !important;
            }
            
            /* Fix dataframe */
            .dataframe, .stDataFrame {
                background: #1e293b !important;
                border: 1px solid #334155 !important;
                border-radius: 16px !important;
            }
            .dataframe th {
                background: #0f172a !important;
                color: white !important;
            }
            .dataframe td {
                color: #cbd5e1 !important;
            }
            
            /* ========== FOOTER - BOTTOM RIGHT ========== */
            .footer {
                position: fixed !important;
                bottom: 0 !important;
                right: 0 !important;
                left: auto !important;
                top: auto !important;
                background: #0f172a !important;
                padding: 6px 16px !important;
                border-top: 1px solid #334155 !important;
                border-left: 1px solid #334155 !important;
                border-radius: 12px 0 0 0 !important;
                z-index: 999 !important;
                font-size: 0.7rem !important;
                text-align: right !important;
            }
            
            .footer div {
                display: flex !important;
                align-items: center !important;
                justify-content: flex-end !important;
                gap: 6px !important;
                flex-wrap: wrap !important;
            }
            
            /* Space for footer */
            .main .block-container {
                padding-bottom: 60px !important;
            }
            
            /* Sidebar */
            [data-testid="stSidebar"] {
                background: #0f172a !important;
                border-right: 1px solid #1e293b !important;
            }
            [data-testid="stSidebar"] * {
                color: #e2e8f0 !important;
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background: #1e293b !important;
                border-radius: 12px !important;
                padding: 6px !important;
            }
            .stTabs [data-baseweb="tab"] {
                background: #0f172a !important;
                color: #94a3b8 !important;
                border-radius: 8px !important;
                padding: 10px 20px !important;
            }
            .stTabs [aria-selected="true"] {
                background: #3b82f6 !important;
                color: white !important;
            }
            .stTabs [data-baseweb="tab-panel"] {
                background: transparent !important;
                padding-top: 20px !important;
            }
            
            /* Text */
            .stMarkdown, .stMarkdown p, p, li, label {
                color: #e2e8f0 !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: white !important;
            }
            
            /* Buttons */
            .stButton > button {
                background: #3b82f6 !important;
                color: white !important;
                border-radius: 40px !important;
            }
            
            /* Inputs */
            .stTextInput input, .stTextArea textarea {
                background: #1e293b !important;
                border: 1px solid #334155 !important;
                color: #e2e8f0 !important;
            }
            
            /* Hero */
            .hero {
                background: linear-gradient(135deg, #0f172a, #1e293b) !important;
                border: 1px solid #334155 !important;
            }
            .hero h1 {
                color: white !important;
            }
            .hero p {
                color: #94a3b8 !important;
            }
            
            /* DNA Background */
            .dna-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: -1;
                pointer-events: none;
                opacity: 0.08;
            }
            .dna-bg::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cpath fill='none' stroke='%233b82f6' stroke-width='1' d='M30,30 L170,170 M170,30 L30,170 M50,10 L150,190 M150,10 L50,190 M70,0 L130,200 M130,0 L70,200'/%3E%3C/svg%3E");
                background-repeat: repeat;
                background-size: 60px 60px;
                animation: slowMove 60s linear infinite;
            }
            @keyframes slowMove {
                0% { transform: translateX(0) translateY(0); }
                100% { transform: translateX(100px) translateY(100px); }
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: #1e293b;
            }
            ::-webkit-scrollbar-thumb {
                background: #3b82f6;
                border-radius: 4px;
            }
            
            /* FAB Button */
            .fab {
                position: fixed;
                bottom: 30px;
                right: 30px;
                background: linear-gradient(135deg, #3b82f6, #2563eb);
                color: white;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                z-index: 1000;
                font-size: 24px;
                text-decoration: none;
            }
            
            @media (max-width: 768px) {
                .hero h1 { font-size: 1.8rem; }
                .metric-value { font-size: 1.2rem; }
                .card { padding: 12px; }
                .hero { padding: 20px; }
            }
        </style>
        <div class="dna-bg"></div>
        <a href="#" class="fab">↑</a>
        """
else:
    if st.session_state.reading_mode:
        theme_css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Georgia', 'Times New Roman', serif; }
            .stApp { background: #fef9e8; }
            p, li, .stMarkdown { font-size: 1.1rem !important; line-height: 1.8 !important; color: #2c3e50 !important; }
            h1 { font-size: 2.5rem !important; color: #2c3e50 !important; }
            h2 { font-size: 2rem !important; color: #2c3e50 !important; }
            h3 { font-size: 1.5rem !important; color: #2c3e50 !important; }
            .card, .metric-card { background: #fffaf0 !important; border: 1px solid #e2d5c0; }
            .hero { background: #f5e6d3; border: 1px solid #d4b88c; }
            .footer { position: fixed; bottom: 0; right: 0; background: #f5e6d3; padding: 6px 16px; border-top: 1px solid #d4b88c; border-left: 1px solid #d4b88c; border-radius: 12px 0 0 0; z-index: 999; font-size: 0.7rem; }
            .main .block-container { padding-bottom: 60px; }
        </style>
        """
    else:
        theme_css = """
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', -apple-system, sans-serif; }
            .stApp { background: #f8fafc; }
            .dna-bg { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; opacity: 0.05; }
            .dna-bg::before { content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'%3E%3Cpath fill='none' stroke='%233b82f6' stroke-width='1' d='M30,30 L170,170 M170,30 L30,170 M50,10 L150,190 M150,10 L50,190 M70,0 L130,200 M130,0 L70,200'/%3E%3C/svg%3E"); background-repeat: repeat; background-size: 60px 60px; animation: slowMove 60s linear infinite; }
            @keyframes slowMove { 0% { transform: translateX(0) translateY(0); } 100% { transform: translateX(100px) translateY(100px); } }
            [data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0; }
            .hero { background: linear-gradient(135deg, #f1f5f9, #e2e8f0); border-radius: 24px; padding: 40px 32px; margin-bottom: 32px; }
            .hero h1 { font-size: 3rem; font-weight: 700; color: #0f172a; }
            .card { background: white; border-radius: 20px; padding: 24px; border: 1px solid #e2e8f0; }
            .metric-card { background: white; border-radius: 20px; padding: 20px; text-align: center; border: 1px solid #e2e8f0; }
            .metric-value { font-size: 2rem; font-weight: 700; color: #0f172a; }
            .footer { position: fixed; bottom: 0; right: 0; background: white; padding: 6px 16px; border-top: 1px solid #e2e8f0; border-left: 1px solid #e2e8f0; border-radius: 12px 0 0 0; z-index: 999; font-size: 0.7rem; }
            .main .block-container { padding-bottom: 60px; }
            .stButton > button { background: #3b82f6 !important; color: white !important; border-radius: 40px !important; }
            .fab { position: fixed; bottom: 30px; right: 30px; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; z-index: 1000; font-size: 24px; text-decoration: none; }
            @media (max-width: 768px) { .hero h1 { font-size: 1.8rem; } .metric-value { font-size: 1.2rem; } .card { padding: 12px; } }
        </style>
        <div class="dna-bg"></div>
        <a href="#" class="fab">↑</a>
        """

st.markdown(theme_css, unsafe_allow_html=True)

# ============================================
# LOAD DATA & MODELS
# ============================================
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    possible_paths = [
        os.path.join(base_dir, "models", "neural_network_model.keras"),
        os.path.join(base_dir, "scaler.pkl"),
        os.path.join(parent_dir, "results", "neural_network_model.keras"),
        os.path.join(parent_dir, "results", "scaler.pkl"),
    ]
    
    model_path = None
    scaler_path = None
    
    for i in range(0, len(possible_paths)-1, 2):
        if os.path.exists(possible_paths[i]) and os.path.exists(possible_paths[i+1]):
            model_path = possible_paths[i]
            scaler_path = possible_paths[i+1]
            break
    
    if model_path and scaler_path:
        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except:
            return None, None
    return None, None

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    possible_paths = [
        os.path.join(base_dir, "data", "smiles_generated.csv"),
        os.path.join(parent_dir, "results", "smiles_generated.csv"),
        os.path.join(parent_dir, "results", "new_molecules_predictions.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df
            except:
                continue
    
    return pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC'],
        'MolWt': [46.07, 45.08, 44.09],
        'LogP': [0.23, 0.15, 0.25],
        'NumHDonors': [1, 1, 1],
        'NumHAcceptors': [1, 1, 1],
        'NumRotatableBonds': [0, 0, 0],
        'NumAromaticRings': [0, 0, 0],
        'TPSA': [20.23, 20.23, 20.23],
        'FractionCSP3': [0.0, 0.0, 0.0],
        'predicted_ic50_nM': [10.5, 12.3, 8.7]
    })

@st.cache_data
def load_admet_results():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    admet_path = os.path.join(parent_dir, "results", "admet_results.csv")
    if os.path.exists(admet_path):
        try:
            return pd.read_csv(admet_path)
        except:
            pass
    return None

@st.cache_data
def load_new_molecules():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    new_mols_path = os.path.join(parent_dir, "results", "new_molecules_predictions.csv")
    if os.path.exists(new_mols_path):
        try:
            return pd.read_csv(new_mols_path)
        except:
            pass
    return None

@st.cache_data
def load_lubatinib_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    lubatinib_files = {
        'pdb': os.path.join(base_dir, "results", "lubatinib.pdb"),
        'png': os.path.join(base_dir, "results", "lubatinib.png"),
        'smi': os.path.join(base_dir, "results", "lubatinib.smi"),
        'txt': os.path.join(base_dir, "results", "lubatinib.txt")
    }
    
    data = {}
    for key, path in lubatinib_files.items():
        if os.path.exists(path):
            try:
                if key == 'smi':
                    with open(path, 'r') as f:
                        data[key] = f.read().strip()
                elif key == 'txt':
                    with open(path, 'r') as f:
                        data[key] = f.read()
                else:
                    data[key] = path
            except:
                data[key] = None
        else:
            data[key] = None
    
    # If not found, try parent directory
    if not any(data.values()):
        parent_dir = os.path.dirname(base_dir)
        parent_files = {
            'pdb': os.path.join(parent_dir, "results", "lubatinib.pdb"),
            'png': os.path.join(parent_dir, "results", "lubatinib.png"),
            'smi': os.path.join(parent_dir, "results", "lubatinib.smi"),
            'txt': os.path.join(parent_dir, "results", "lubatinib.txt")
        }
        for key, path in parent_files.items():
            if os.path.exists(path) and data[key] is None:
                try:
                    if key == 'smi':
                        with open(path, 'r') as f:
                            data[key] = f.read().strip()
                    elif key == 'txt':
                        with open(path, 'r') as f:
                            data[key] = f.read()
                    else:
                        data[key] = path
                except:
                    pass
    
    return data

# Show loading animation with progress
with st.spinner(""):
    st.markdown(show_loading_animation(), unsafe_allow_html=True)
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Initializing SEGULAH AI...")
    
    for i in range(101):
        time.sleep(0.005)
        progress_bar.progress(i / 100, text=f"Loading SEGULAH AI... {i}%")
    
    model, scaler = load_model()
    df = load_data()
    df_admet = load_admet_results()
    df_new_molecules = load_new_molecules()
    lubatinib = load_lubatinib_data()
    
    progress_placeholder.empty()

# Add ADMET Score
def calculate_admet_score(row):
    score = 0
    if row.get('MolWt', 0) < 500: score += 25
    if row.get('LogP', 0) < 5: score += 25
    if row.get('NumHDonors', 0) < 5: score += 25
    if row.get('NumHAcceptors', 0) < 10: score += 25
    return score

if not df.empty:
    df['ADMET_Score'] = df.apply(calculate_admet_score, axis=1)
    df['Potency_Class'] = pd.cut(df['predicted_ic50_nM'], 
                                   bins=[0, 1, 10, 100, 1000, float('inf')],
                                   labels=['Super Potent', 'Highly Potent', 'Potent', 'Moderate', 'Weak'])

add_notification("SEGULAH AI ready!", "success", push=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def calculate_drug_likeness(mol):
    if mol is None:
        return 0
    score = 0
    if Descriptors.MolWt(mol) <= 500: score += 20
    if Descriptors.MolLogP(mol) <= 5: score += 20
    if Descriptors.NumHDonors(mol) <= 5: score += 20
    if Descriptors.NumHAcceptors(mol) <= 10: score += 20
    if Descriptors.NumRotatableBonds(mol) <= 10: score += 10
    if Descriptors.TPSA(mol) <= 140: score += 10
    return score

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.button("🌙 Toggle Dark Mode", on_click=toggle_theme, key="theme_btn", use_container_width=True)
    st.button("📖 Toggle Reading Mode", on_click=toggle_reading_mode, key="reading_btn", use_container_width=True)
    
    if not st.session_state.push_notifications_enabled:
        if st.button("🔔 Enable Push Notifications", use_container_width=True):
            enable_push_notifications()
            add_notification("Push notifications enabled!", "success")
    else:
        st.success("🔔 Notifications enabled")
    
    logo_html = get_logo_html_transparent(40)
    st.markdown(f'<div style="text-align: center; margin: 10px 0;">{logo_html}</div>', unsafe_allow_html=True)
    
    st.markdown("### SEGULAH AI")
    st.markdown("*Intelligent Drug Discovery*")
    st.markdown("---")
    
    option = st.selectbox(
        "🔬 Navigation",
        [" Dashboard", " Predict IC50", " Generate Molecules", " Drug Formulation", 
         " Lubatinib", " Comparative Analysis", " Mutation Analysis", " Docking Viewer",
         " Reports", " API Integration", " View Molecules", " About"],
        index=0,
        key="navigation_select"
    )
    st.session_state.current_option = option
    
    st.markdown("---")
    st.markdown("### 📊 Platform Stats")
    if not df.empty:
        st.metric("Total Molecules", f"{len(df):,}")
        st.metric("Best IC50", f"{df['predicted_ic50_nM'].min():.3f} nM")
        st.metric("Drug-like", f"{len(df[df['ADMET_Score'] >= 75])} molecules")
    
    if lubatinib.get('smi'):
        st.markdown("---")
        st.markdown("### ⭐ Featured")
        st.success("**Lubatinib**\nNovel EGFR Inhibitor")

# ============================================
# HERO SECTION
# ============================================
logo_large = get_logo_transparent(120)
if logo_large:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(logo_large, width=120)
    with col2:
        st.markdown("""
        <div class="hero" style="margin-left: 0;">
            <h1>⚛️ SEGULAH AI</h1>
            <p>AI-Powered Drug Discovery Platform | Targeting EGFR | Complete Drug Development Suite</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="hero">
        <h1>⚛️ SEGULAH AI</h1>
        <p>AI-Powered Drug Discovery Platform | Targeting EGFR | Complete Drug Development Suite</p>
    </div>
    """, unsafe_allow_html=True)

if not st.session_state.notifications_shown:
    show_notifications()
    st.session_state.notifications_shown = True

# ============================================
# DASHBOARD
# ============================================
if option == " Dashboard":
    st.markdown("## Dashboard")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Molecules</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['predicted_ic50_nM'].min():.3f}</div>
            <div class="metric-label">Best IC50 (nM)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['ADMET_Score'].mean():.0f}</div>
            <div class="metric-label">Avg ADMET Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">0.620</div>
            <div class="metric-label">Model R²</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Advanced Analytics Row
    st.markdown("### 📊 Advanced Analytics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_drug_score = df['ADMET_Score'].mean()
        st.metric("Avg Drug Score", f"{avg_drug_score:.0f}/100")
    
    with col2:
        potent_count = len(df[df['predicted_ic50_nM'] < 10])
        st.metric("Potent Molecules", potent_count)
    
    with col3:
        drug_like_count = len(df[df['ADMET_Score'] >= 75])
        st.metric("Drug-like", f"{drug_like_count}/{len(df)}")
    
    with col4:
        st.metric("Reports Generated", st.session_state.reports_generated)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Potency Distribution")
        potency_counts = df['Potency_Class'].value_counts().reset_index()
        potency_counts.columns = ['Class', 'Count']
        fig = px.pie(potency_counts, values='Count', names='Class', color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, key="pie_dash")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 IC50 Distribution")
        fig = px.histogram(df, x='predicted_ic50_nM', nbins=50, color_discrete_sequence=['#3b82f6'])
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, key="hist_dash")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Prediction Timeline
    if st.session_state.prediction_history:
        with st.expander("📈 Prediction Timeline", expanded=False):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            fig = px.line(history_df, x='timestamp', y='ic50', title="IC50 Predictions Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    # New Molecules Section
    if df_new_molecules is not None and not df_new_molecules.empty:
        with st.expander("🧪 New Generated Molecules", expanded=False):
            st.markdown(f"**Total new molecules:** {len(df_new_molecules)}")
            if 'predicted_ic50_nM' in df_new_molecules.columns:
                best_new = df_new_molecules.nsmallest(5, 'predicted_ic50_nM')
                st.markdown("#### 🏆 Top 5 New Molecules")
                st.dataframe(best_new, use_container_width=True)
    
    # ADMET Section
    if df_admet is not None and not df_admet.empty:
        with st.expander("📊 ADMET Analysis Results", expanded=False):
            st.dataframe(df_admet.head(10), use_container_width=True)
    
    # Lubatinib Featured
    if lubatinib.get('smi'):
        st.markdown("---")
        st.markdown("## ⭐ Featured: Lubatinib")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if lubatinib.get('png') and os.path.exists(lubatinib['png']):
                st.image(lubatinib['png'], width=120)
        with col2:
            st.markdown("**Lubatinib** - Novel EGFR Tyrosine Kinase Inhibitor")
            st.markdown("Designed via CB-Dock molecular docking")
            mol = Chem.MolFromSmiles(lubatinib['smi'])
            if mol:
                drug_score = calculate_drug_likeness(mol)
                st.progress(drug_score/100)
                st.caption(f"Drug Score: {drug_score}/100")
        with col3:
            # Direct extraction from file - gets all cavities and picks best
            txt_content = lubatinib.get('txt', '')
            binding_energy = None
            cavity_volume = None
            cavity_id = None
            all_results = []
            
            if txt_content:
                import re
                lines = txt_content.strip().split('\n')
                
                # Collect all cavities
                for line in lines:
                    match = re.match(r'^(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.?\d*)', line)
                    if match:
                        try:
                            cid = int(match.group(1))
                            cvol = float(match.group(2))
                            cscore = float(match.group(9))
                            all_results.append({
                                'id': cid,
                                'volume': cvol,
                                'score': cscore
                            })
                        except:
                            pass
                
                # Pick best cavity (lowest score = most negative)
                if all_results:
                    best = min(all_results, key=lambda x: x['score'])
                    cavity_id = best['id']
                    cavity_volume = best['volume']
                    binding_energy = best['score']
            
            if binding_energy is not None:
                st.metric("Binding Energy", f"{binding_energy} kcal/mol")
                st.metric("Cavity Volume", f"{cavity_volume} Å³")
                st.metric("Cavity ID", cavity_id)
            else:
                st.info("No docking results available")

# ============================================
# PREDICT IC50
# ============================================
elif option == " Predict IC50":
    st.markdown("## 🔮 Predict IC50 from SMILES")
    st.markdown("---")
    
    if model is None or scaler is None:
        st.warning("⚠️ Model not loaded. Using simplified prediction.")
        
        with st.form(key="predict_form_simple"):
            smiles = st.text_area("Enter SMILES string:", "c1ccc2ccccc2c1COCOCOCS(=O)C", height=100)
            submitted = st.form_submit_button("🔮 Predict IC50", type="primary", use_container_width=True)
            
            if submitted and smiles.strip():
                show_operation_progress("Analyzing molecule", 1.5)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    st.info("📊 Demo mode: Model not available. Showing sample prediction.")
                    st.success(f"### 🎯 Predicted IC50: **~15.2 nM**")
                    add_notification("Demo prediction completed", "info", push=True)
                else:
                    st.error("❌ Invalid SMILES string!")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            with st.form(key="predict_form"):
                smiles = st.text_area("Enter SMILES string:", "c1ccc2ccccc2c1COCOCOCS(=O)C", height=100)
                submitted = st.form_submit_button("🔮 Predict IC50", type="primary", use_container_width=True)
                
                if submitted and smiles.strip():
                    show_operation_progress("Analyzing molecule", 1.5)
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        features = [
                            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
                            Descriptors.TPSA(mol), Descriptors.FractionCSP3(mol)
                        ]
                        features_scaled = scaler.transform([features])
                        pred_log = model.predict(features_scaled, verbose=0)[0][0]
                        pred_ic50 = 10 ** pred_log
                        
                        st.success(f"### 🎯 Predicted IC50: **{pred_ic50:.3f} nM**")
                        add_notification(f"Predicted IC50: {pred_ic50:.3f} nM", "success", push=True)
                        
                        if pred_ic50 < 5:
                            show_balloons_celebration()
                            show_toast_message("🌟 Excellent! Super potent molecule!", "🎉")
                            show_confetti()
                        elif pred_ic50 < 10:
                            show_toast_message("👍 Good potency! Promising candidate", "💊")
                            show_confetti()
                        
                        violations = sum([features[0] > 500, features[1] > 5, features[2] > 5, features[3] > 10])
                        admet_score = 100 - (violations * 25)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if violations == 0:
                                st.success("✅ Lipinski: Passed (0 violations)")
                            else:
                                st.warning(f"⚠️ Lipinski: {violations} violations")
                        with col_b:
                            st.metric("ADMET Score", f"{admet_score}/100")
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'smiles': smiles[:50] + "..." if len(smiles) > 50 else smiles,
                            'ic50': pred_ic50,
                            'admet': admet_score
                        })
                        
                        with st.expander("🔬 Advanced Properties"):
                            sa_score = calculate_synthetic_accessibility(mol)
                            bbb = calculate_bbb_permeability(mol)
                            quantum = calculate_quantum_properties(mol)
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Synthetic Accessibility", f"{sa_score:.1f}/10")
                                st.metric("BBB Permeability", bbb)
                            with col_b:
                                st.metric("HOMO (eV)", f"{quantum['HOMO_Estimate (eV)']:.2f}")
                                st.metric("LUMO (eV)", f"{quantum['LUMO_Estimate (eV)']:.2f}")
                            with col_c:
                                st.metric("Band Gap (eV)", f"{quantum['Band_Gap (eV)']:.2f}")
                                st.metric("Dipole Moment", f"{quantum['Dipole_Moment (D)']:.2f} D")
                    else:
                        st.error("❌ Invalid SMILES string!")
        
        with col2:
            st.markdown("### 💡 Example SMILES")
            st.code("c1ccc2ccccc2c1COCOCOCS(=O)C", language="python")
            st.caption("Molecule #0 - ~5.86 nM")
            if lubatinib.get('smi'):
                st.markdown("---")
                st.markdown("### ⭐ Lubatinib")
                st.code(lubatinib['smi'][:60] + "..." if len(lubatinib['smi']) > 60 else lubatinib['smi'], language="python")
                st.caption("Novel EGFR Inhibitor")

# ============================================
# GENERATE MOLECULES
# ============================================
elif option == " Generate Molecules":
    st.markdown("## 🧪 Generate Molecules from Properties")
    st.markdown("---")
    
    if model is None or scaler is None:
        st.warning("⚠️ Model not loaded. Using demo mode.")
        
        with st.form(key="generate_form_demo"):
            col1, col2 = st.columns(2)
            with col1:
                molwt = st.slider("Molecular Weight (g/mol)", 200, 600, 400)
                logp = st.slider("LogP", -2.0, 5.0, 2.5, step=0.1)
                h_donors = st.slider("H-Bond Donors", 0, 5, 2)
                h_acceptors = st.slider("H-Bond Acceptors", 0, 10, 5)
            with col2:
                rot_bonds = st.slider("Rotatable Bonds", 0, 12, 5)
                aromatic_rings = st.slider("Aromatic Rings", 0, 4, 2)
                tpsa = st.slider("TPSA (Å²)", 20, 140, 80)
                fraction_csp3 = st.slider("Fraction CSP3", 0.0, 1.0, 0.5, step=0.05)
            
            submitted = st.form_submit_button("🎯 Predict IC50", type="primary", use_container_width=True)
            
            if submitted:
                show_operation_progress("Generating prediction", 1.5)
                st.info("📊 Demo mode: Model not available. Showing sample prediction.")
                pred_ic50 = 15.2
                st.success(f"### 🎯 Predicted IC50: **{pred_ic50:.3f} nM**")
    else:
        with st.form(key="generate_form"):
            col1, col2 = st.columns(2)
            with col1:
                molwt = st.slider("Molecular Weight (g/mol)", 200, 600, 400)
                logp = st.slider("LogP", -2.0, 5.0, 2.5, step=0.1)
                h_donors = st.slider("H-Bond Donors", 0, 5, 2)
                h_acceptors = st.slider("H-Bond Acceptors", 0, 10, 5)
            with col2:
                rot_bonds = st.slider("Rotatable Bonds", 0, 12, 5)
                aromatic_rings = st.slider("Aromatic Rings", 0, 4, 2)
                tpsa = st.slider("TPSA (Å²)", 20, 140, 80)
                fraction_csp3 = st.slider("Fraction CSP3", 0.0, 1.0, 0.5, step=0.05)
            
            submitted = st.form_submit_button("🎯 Predict IC50", type="primary", use_container_width=True)
            
            if submitted:
                show_operation_progress("Running AI prediction", 1.5)
                features = [[molwt, logp, h_donors, h_acceptors, rot_bonds, aromatic_rings, tpsa, fraction_csp3]]
                features_scaled = scaler.transform(features)
                pred_log = model.predict(features_scaled, verbose=0)[0][0]
                pred_ic50 = 10 ** pred_log
                
                st.success(f"### 🎯 Predicted IC50: **{pred_ic50:.3f} nM**")
                add_notification(f"Generated molecule with IC50: {pred_ic50:.3f} nM", "success", push=True)
                
                if pred_ic50 < 10:
                    show_balloons_celebration()
                    show_confetti()
                
                violations = sum([molwt > 500, logp > 5, h_donors > 5, h_acceptors > 10])
                admet_score = 100 - (violations * 25)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.success("✅ Lipinski: Passed" if violations == 0 else f"⚠️ {violations} violations")
                with col_b:
                    st.info("📊 Potency: High" if pred_ic50 < 10 else "Medium" if pred_ic50 < 100 else "Low")
                with col_c:
                    st.metric("ADMET Score", f"{admet_score}/100")
                
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'molwt': molwt, 'logp': logp, 'ic50': pred_ic50, 'admet': admet_score
                })

# ============================================
# DRUG FORMULATION
# ============================================
elif option == " Drug Formulation":
    st.markdown("## 💊 Drug Formulation & Analysis")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Compound Analysis", "💊 PK/PD", "⚠️ Toxicity", "🧬 Mutations"])
    
    with tab1:
        with st.form(key="formulation_form"):
            smiles_form = st.text_area("Enter SMILES:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", height=80)
            analyze = st.form_submit_button("🔬 Analyze Compound", type="primary", use_container_width=True)
            
            if analyze and smiles_form:
                show_operation_progress("Analyzing compound", 1.2)
                mol = Chem.MolFromSmiles(smiles_form)
                if mol:
                    st.success("✅ Valid SMILES structure")
                    props = {
                        "Molecular Weight": f"{Descriptors.MolWt(mol):.2f} g/mol",
                        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                        "H-Bond Donors": Descriptors.NumHDonors(mol),
                        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                    }
                    st.dataframe(pd.DataFrame(list(props.items()), columns=['Property', 'Value']))
                    drug_score = calculate_drug_likeness(mol)
                    st.progress(drug_score/100)
                    st.caption(f"Drug Score: {drug_score}/100")
                    
                    img = Draw.MolToImage(mol, size=(300, 300))
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.image(buf.getvalue(), width=300)
    
    with tab2:
        smiles_pk = st.text_input("SMILES for PK/PD:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        if smiles_pk:
            mol = Chem.MolFromSmiles(smiles_pk)
            if mol:
                pk_params = predict_pk_parameters(mol)
                for key, val in pk_params.items():
                    st.metric(key, f"{val:.1f}")
    
    with tab3:
        smiles_tox = st.text_input("SMILES for Toxicity:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        if smiles_tox:
            mol = Chem.MolFromSmiles(smiles_tox)
            if mol:
                toxicity = predict_toxicity(mol)
                for key, val in toxicity.items():
                    color = "green" if "Low" in str(val) else "orange" if "Medium" in str(val) else "red"
                    st.markdown(f"**{key}:** <span style='color:{color}'>{val}</span>", unsafe_allow_html=True)
    
    with tab4:
        smiles_mut = st.text_input("SMILES for Mutation:", lubatinib.get('smi', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'))
        if smiles_mut:
            resistance = get_mutation_resistance(smiles_mut)
            for mut, score in resistance.items():
                st.progress(score/100)
                st.caption(f"{mut}: {score:.1f}% resistance risk")

# ============================================
# LUBATINIB
# ============================================
elif option == " Lubatinib":
    st.markdown("## ⭐ Lubatinib - Novel EGFR Inhibitor")
    st.markdown("---")
    
    if lubatinib.get('smi'):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 20px; padding: 30px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0;">⚛️ Lubatinib</h1>
            <p style="color: rgba(255,255,255,0.9);">A Novel EGFR Tyrosine Kinase Inhibitor Designed via CB-Dock</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.code(lubatinib['smi'], language="python")
            mol = Chem.MolFromSmiles(lubatinib['smi'])
            if mol:
                props = {"MW": f"{Descriptors.MolWt(mol):.2f}", "LogP": f"{Descriptors.MolLogP(mol):.2f}"}
                for k, v in props.items():
                    st.metric(k, v)
                drug_score = calculate_drug_likeness(mol)
                st.progress(drug_score/100)
        
        with col2:
            if lubatinib.get('png') and os.path.exists(lubatinib['png']):
                st.image(lubatinib['png'], width=350)
            elif mol:
                img = Draw.MolToImage(mol, size=(350, 350))
                buf = BytesIO()
                img.save(buf, format="PNG")
                st.image(buf.getvalue(), width=350)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Docking", "💊 PK/PD", "⚠️ Toxicity", "🧬 Mutations"])
        
        with tab1:
            st.subheader("🎯 Docking Results")
            
            # Direct extraction from file - gets all cavities and picks best
            txt_content = lubatinib.get('txt', '')
            binding_energy = None
            cavity_volume = None
            cavity_id = None
            rmsd = None
            all_results = []
            
            if txt_content:
                import re
                lines = txt_content.strip().split('\n')
                
                # Collect all cavities
                for line in lines:
                    match = re.match(r'^(\d+)\s+(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.?\d*)', line)
                    if match:
                        try:
                            cid = int(match.group(1))
                            cvol = float(match.group(2))
                            cscore = float(match.group(9))
                            all_results.append({
                                'id': cid,
                                'volume': cvol,
                                'score': cscore
                            })
                        except:
                            pass
                
                # Pick best cavity (lowest score = most negative)
                if all_results:
                    best = min(all_results, key=lambda x: x['score'])
                    cavity_id = best['id']
                    cavity_volume = best['volume']
                    binding_energy = best['score']
                    
                    # Show info about all cavities
                    if len(all_results) > 1:
                        st.info(f"📊 Found {len(all_results)} cavities. Best is Cavity {cavity_id} (Score: {binding_energy} kcal/mol)")
                
                # Look for RMSD in various formats
                # Pattern 1: "RMSD: < 2.0 Å" or "RMSD: 1.2"
                rmsd_match = re.search(r'RMSD:\s*([<\d\.]+\s*\d*\.?\d*)\s*Å', txt_content, re.IGNORECASE)
                if rmsd_match:
                    rmsd = rmsd_match.group(1).strip()
                else:
                    # Pattern 2: "RMSD: 1.2" without Å
                    rmsd_match2 = re.search(r'RMSD:\s*([<\d\.]+\s*\d*\.?\d*)', txt_content, re.IGNORECASE)
                    if rmsd_match2:
                        rmsd = rmsd_match2.group(1).strip()
                    else:
                        # Pattern 3: Look for RMSD in the best cavity line
                        for line in lines:
                            if str(cavity_id) in line and 'RMSD' in line:
                                rmsd_values = re.findall(r'([<\d\.]+\s*\d*\.?\d*)', line)
                                if rmsd_values:
                                    rmsd = rmsd_values[0].strip()
                                    break
            
            # Display results
            if binding_energy is not None:
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Binding Energy", f"{binding_energy} kcal/mol")
                with col_b:
                    st.metric("Cavity Volume", f"{cavity_volume} Å³")
                with col_c:
                    st.metric("Cavity ID", cavity_id)
                with col_d:
                    if rmsd is not None:
                        st.metric("RMSD", f"{rmsd} Å")
                    else:
                        st.metric("RMSD", "N/A")
                
                if binding_energy < 0:
                    st.success(f"✅ Good binding affinity: {binding_energy} kcal/mol")
                
                # Show all cavities in expander
                if all_results and len(all_results) > 1:
                    with st.expander("🔍 View all cavities"):
                        cavity_df = pd.DataFrame(all_results)
                        cavity_df = cavity_df.sort_values('score')
                        cavity_df.columns = ['Cavity ID', 'Volume (Å³)', 'Score (kcal/mol)']
                        st.dataframe(cavity_df, use_container_width=True)
            else:
                st.info("📌 No docking results available. Please check lubatinib.txt file.")
                if lubatinib.get('txt'):
                    with st.expander("🔍 View lubatinib.txt content"):
                        st.code(lubatinib['txt'][:1000], language="text")
        
        with tab2:
            if mol:
                pk_params = predict_pk_parameters(mol)
                for key, val in pk_params.items():
                    st.metric(key, f"{val:.1f}")
        
        with tab3:
            if mol:
                toxicity = predict_toxicity(mol)
                for key, val in toxicity.items():
                    color = "green" if "Low" in str(val) else "orange" if "Medium" in str(val) else "red"
                    st.markdown(f"**{key}:** <span style='color:{color}'>{val}</span>", unsafe_allow_html=True)
        
        with tab4:
            resistance = get_mutation_resistance(lubatinib['smi'])
            for mut, score in resistance.items():
                st.progress(score/100)
                st.caption(f"{mut}: {score:.1f}%")

# ============================================
# COMPARATIVE ANALYSIS
# ============================================
elif option == " Comparative Analysis":
    st.markdown("## 📊 Comparative Drug Analysis")
    st.markdown("---")
    
    known_drugs = {
        "Gefitinib": "COc1cc2c(cc1OCCCN3CCOCC3)ncnc2Nc4ccc(F)c(Cl)c4",
        "Erlotinib": "COc1cc2ncnc(Nc3ccc(OCCOC)c(OCCOC)c3)c2cc1OCCOC",
        "Osimertinib": "CN1C=C(C2=C1C=NC(=C2)NC3=C(C=C(C(=C3)NC(=O)C=C)N(C)CCN(C)C)F)C(=O)NC",
        "Lubatinib": lubatinib.get('smi', '')
    }
    
    compounds_to_compare = st.multiselect("Select drugs:", list(known_drugs.keys()), default=["Gefitinib", "Lubatinib"])
    
    if compounds_to_compare:
        comparison_data = []
        for drug in compounds_to_compare:
            smiles = known_drugs.get(drug, '')
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    comparison_data.append({
                        'Drug': drug,
                        'MolWt': round(Descriptors.MolWt(mol), 2),
                        'LogP': round(Descriptors.MolLogP(mol), 2),
                        'Drug Score': calculate_drug_likeness(mol)
                    })
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# ============================================
# MUTATION ANALYSIS
# ============================================
elif option == " Mutation Analysis":
    st.markdown("## 🧬 EGFR Mutation Resistance Analysis")
    st.markdown("---")
    
    smiles_mut = st.text_input("Enter SMILES:", lubatinib.get('smi', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'))
    
    if smiles_mut:
        mutations = ['T790M', 'L858R', 'C797S', 'G719S', 'L861Q']
        resistance = get_mutation_resistance(smiles_mut, mutations)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(resistance.keys()), y=list(resistance.values()), marker_color='#ef4444'))
        fig.update_layout(title="Resistance Score", yaxis_title="Score (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# DOCKING VIEWER
# ============================================
elif option == " Docking Viewer":
    st.markdown("## 🔬 Molecular Docking Viewer")
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 20px; padding: 25px; margin-bottom: 25px;">
        <h2 style="color: white; margin: 0;">⚛️ Interactive 3D Docking Viewer</h2>
        <p style="color: rgba(255,255,255,0.9);">Visualize protein-ligand interactions in 3D</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not DOCKING_AVAILABLE:
        st.warning("⚠️ py3Dmol not installed. Run: pip install py3Dmol")
        st.info("For now, you can view the structure in 2D:")
        
        if lubatinib.get('png') and os.path.exists(lubatinib['png']):
            st.image(lubatinib['png'], width=400, caption="Lubatinib 2D Structure")
        elif lubatinib.get('smi'):
            mol = Chem.MolFromSmiles(lubatinib['smi'])
            if mol:
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, caption="Lubatinib Structure")
    else:
        pdb_files = get_available_pdb_files()
        
        if not pdb_files:
            st.warning("⚠️ No PDB files found. Please add protein structure files.")
            if lubatinib.get('pdb') and os.path.exists(lubatinib['pdb']):
                with open(lubatinib['pdb'], 'r') as f:
                    pdb_content = f.read()
                st.download_button("📥 Download Lubatinib PDB", pdb_content, "lubatinib.pdb", "chemical/x-pdb")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_file = st.selectbox("Select Protein Structure:", pdb_files, format_func=lambda x: os.path.basename(x))
            
            with col2:
                ligand_options = ["None", "Lubatinib"]
                if lubatinib.get('pdb') and os.path.exists(lubatinib['pdb']):
                    selected_ligand = st.selectbox("Select Ligand:", ligand_options)
                else:
                    selected_ligand = "None"
                    st.info("Lubatinib PDB not found")
            
            if selected_file:
                try:
                    with open(selected_file, 'r') as f:
                        pdb_content = f.read()
                    
                    view = py3Dmol.view(width=800, height=500)
                    view.addModel(pdb_content, 'pdb')
                    view.setStyle({'cartoon': {'color': 'spectrum'}})
                    
                    if selected_ligand == "Lubatinib" and lubatinib.get('pdb') and os.path.exists(lubatinib['pdb']):
                        with open(lubatinib['pdb'], 'r') as f:
                            ligand_content = f.read()
                        view.addModel(ligand_content, 'pdb')
                        view.setStyle({'model': 1}, {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
                    
                    view.setBackgroundColor('0x000000' if st.session_state.dark_mode else '0xffffff')
                    view.zoomTo()
                    
                    viewer_html = view._make_html()
                    components.html(viewer_html, height=550, width=850)
                    
                    st.success("✅ 3D structure loaded! Use mouse to rotate/zoom.")
                    
                    st.download_button("📥 Download PDB", pdb_content, os.path.basename(selected_file), "chemical/x-pdb")
                    
                except Exception as e:
                    st.error(f"Error loading viewer: {e}")
            
            with st.expander("📖 Controls"):
                st.markdown("""
                - **Rotate**: Left click + drag
                - **Zoom**: Right click + drag or scroll
                - **Pan**: Shift + left click + drag
                - **Styles**: Use buttons below the viewer
                """)

# ============================================
# REPORTS
# ============================================
elif option == " Reports":
    st.markdown("## 📄 Auto-Generated Drug Reports")
    st.markdown("---")
    
    report_compound = st.selectbox("Select compound:", ["Lubatinib", "Gefitinib", "Erlotinib", "Custom SMILES"])
    
    if report_compound == "Custom SMILES":
        custom_smiles = st.text_area("Enter SMILES:", "")
        if custom_smiles:
            report_smiles = custom_smiles
            report_name = "Custom_Compound"
        else:
            report_smiles = None
    else:
        known_smiles = {
            "Lubatinib": lubatinib.get('smi', ''),
            "Gefitinib": "COc1cc2c(cc1OCCCN3CCOCC3)ncnc2Nc4ccc(F)c(Cl)c4",
            "Erlotinib": "COc1cc2ncnc(Nc3ccc(OCCOC)c(OCCOC)c3)c2cc1OCCOC",
        }
        report_smiles = known_smiles.get(report_compound, '')
        report_name = report_compound
    
    if report_smiles and st.button("📄 Generate Report", type="primary"):
        with st.spinner("Generating report..."):
            show_operation_progress("Creating report", 1.2)
            mol = Chem.MolFromSmiles(report_smiles)
            if mol:
                properties = {
                    "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
                    "LogP": f"{Descriptors.MolLogP(mol):.2f}",
                    "Drug Score": f"{calculate_drug_likeness(mol)}/100"
                }
                html_report = generate_html_report(report_name, report_smiles, properties, {}, {}, {}, {})
                st.download_button("📥 Download Report", html_report, f"{report_name}_report.html", "text/html")
                st.session_state.reports_generated += 1
                add_notification(f"Report generated for {report_name}", "success", push=True)
                if report_name == "Lubatinib":
                    show_confetti()
            else:
                st.error("Invalid SMILES")

# ============================================
# API INTEGRATION
# ============================================
elif option == " API Integration":
    st.markdown("## 🌐 External API Integration")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 ChEMBL")
        chembl_query = st.text_input("Search:", "erlotinib")
        if chembl_query:
            with st.spinner("Fetching..."):
                results = fetch_chembl_data(chembl_query)
                if results:
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.info("No results")
    
    with col2:
        st.markdown("### 📊 PubChem")
        pubchem_smiles = st.text_input("SMILES:", "COc1cc2ncnc(Nc3ccc(OCCOC)c(OCCOC)c3)c2cc1OCCOC")
        if pubchem_smiles and st.button("Fetch"):
            data = fetch_pubchem_data(pubchem_smiles)
            if data:
                st.json(data)

# ============================================
# VIEW MOLECULES
# ============================================
elif option == " View Molecules":
    st.markdown("## 📊 Generated Molecules")
    st.markdown("---")
    
    if not df.empty:
        with st.expander("🔍 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                min_ic50 = st.number_input("Min IC50", 0.0, 1000.0, 0.0)
                max_ic50 = st.number_input("Max IC50", 0.0, 1000.0, 50.0)
            with col2:
                min_molwt = st.number_input("Min MolWt", 200, 600, 200)
                max_molwt = st.number_input("Max MolWt", 200, 600, 600)
            with col3:
                min_admet = st.slider("Min ADMET", 0, 100, 50)
                potency_filter = st.multiselect("Potency", ['Super Potent', 'Highly Potent', 'Potent', 'Moderate', 'Weak'],
                                                default=['Super Potent', 'Highly Potent'])
        
        filtered = df[(df['predicted_ic50_nM'] >= min_ic50) & (df['predicted_ic50_nM'] <= max_ic50) &
                      (df['MolWt'] >= min_molwt) & (df['MolWt'] <= max_molwt) &
                      (df['ADMET_Score'] >= min_admet) & (df['Potency_Class'].isin(potency_filter))]
        
        st.markdown(f"### Results: {len(filtered)} molecules")
        
        if not filtered.empty:
            st.dataframe(filtered[['predicted_ic50_nM', 'ADMET_Score', 'Potency_Class', 'MolWt', 'LogP']].round(3))
            csv = filtered.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "segulah_molecules.csv", "text/csv")

# ============================================
# ABOUT
# ============================================
elif option == " About":
    st.markdown("## 📈 About SEGULAH AI")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ⚛️ SEGULAH AI - Next-Generation Drug Discovery
        
        **SEGULAH AI** is an advanced AI platform for drug discovery targeting EGFR.
        
        ### 🔬 Features
        
        - ✅ **AI-Powered IC50 Prediction** - R² = 0.620
        - ✅ **Drug Formulation Analysis** - PK/PD, Toxicity
        - ✅ **Lubatinib Integration** - Novel EGFR inhibitor
        - ✅ **Comparative Analysis** - Drug comparison
        - ✅ **Mutation Resistance** - EGFR mutation analysis
        - ✅ **3D Docking Viewer** - Interactive visualization
        - ✅ **Push Notifications** - Real-time alerts
        - ✅ **Reading Mode** - Comfortable reading
        
        ### 📊 Model Performance
        
        | Metric | Value |
        |--------|-------|
        | Test R² | 0.620 |
        | Best IC50 | 5.86 nM |
        
        ### 🛠️ Technologies
        
        Python, TensorFlow, RDKit, Streamlit, Plotly, py3Dmol
        """)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <h3>📊 Stats</h3>
            <ul>
                <li>Total Molecules: {len(df):,}</li>
                <li>Drug-like: {len(df[df['ADMET_Score'] >= 75])}</li>
                <li>Reports: {st.session_state.reports_generated}</li>
            </ul>
            <hr>
            <h3>⭐ Featured</h3>
            <p><strong>Lubatinib</strong><br>Novel EGFR Inhibitor</p>
            <hr>
            <h3>👩‍🔬 Developer</h3>
            <p><strong>LUBANAH H. YOUNES</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown(get_footer(), unsafe_allow_html=True)