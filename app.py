import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA (WAJIB PALING ATAS)
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Obesitas", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CSS BRUTE-FORCE (MEMAKSA TAMPILAN TAB)
# ==========================================
st.markdown("""
<style>
    /* Mengimpor Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

    /* PAKSA BACKGROUND WARNA ABU-ABU KEBIRUAN */
    .stApp { background-color: #f0f4f8 !important; }

    /* ==============================================
       MEMAKSA TAB MENU BERWARNA DAN PUNYA BATAS (TIDAK POLOS)
       ============================================== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important; 
        border-radius: 15px !important;
        padding: 8px !important;
        gap: 12px !important;
        box-shadow: inset 0px 3px 6px rgba(0,0,0,0.05) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        border: 2px solid #cbd5e1 !important; 
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05) !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%) !important;
        border: 2px solid #1A2980 !important;
        box-shadow: 0px 6px 15px rgba(38, 208, 206, 0.5) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* ==============================================
       MEMAKSA KOTAK INPUT LEBIH JELAS
       ============================================== */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 20px !important;
        border-radius: 15px !important;
        border: 2px solid #cbd5e1 !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05) !important;
    }
    .stSlider [data-testid="stMarkdownContainer"] p, 
    .stNumberInput [data-testid="stMarkdownContainer"] p, 
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #1A2980 !important;
        font-weight: 800 !important;
    }

    /* TOMBOL UTAMA */
    button[kind="primary"] {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 15px 30px !important;
        font-weight: 900 !important;
        letter-spacing: 1px !important;
        box-shadow: 0px 8px 20px rgba(255, 75, 43, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI LOADING DATA & CACHING
# ==========================================
@st.cache_resource
def load_fast_model():
    with open('model_ensemble_signifikan.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

@st.cache_data
def load_data():
    return pd.read_excel("KEL.2 obesitas Projek MCL 2.xlsx", sheet_name=0)

try:
    meta_data = load_fast_model()
    model = meta_data['model']           
    encoders = meta_data['encoders']     
    feature_names = meta_data['features']
    classes = meta_data['classes']       
    df_raw = load_data()
except Exception as e:
    st.error("❌ File 'model_ensemble_signifikan.pkl' atau Excel tidak ditemukan!")
    st.stop() 

# ==========================================
# 4. SIDEBAR MENU
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-weight: 900;'>⚙️ Navigasi</h2>", unsafe_allow_html=True)
    lang = st.radio("Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.markdown("### 💧 Target Air Harian" if lang == "Bahasa Indonesia" else "### 💧 Daily Water Target")
    lbl_bb = "Berat Badan Anda (kg)" if lang == "Bahasa Indonesia" else "Your Weight (kg)"
    bb_calc = st.number_input(lbl_bb, 30, 200, 60, key="water_calc")
    
    hidrasi_air = bb_calc * 0.033
    lbl_hidrasi = "Kebutuhan Hidrasi Minimum" if lang == "Bahasa Indonesia" else "Minimum Hydration Need"
    lbl_liter = "Liter/hari" if lang == "Bahasa Indonesia" else "Liters/day"
    
    st.markdown(f"""
    <div style="background: #e2e8f0; padding: 15px; border-radius
