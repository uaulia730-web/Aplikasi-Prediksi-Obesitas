import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Obesitas", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CUSTOM CSS (TEMA PREMIUM & SELARAS)
# ==========================================
st.markdown("""
<style>
    /* Mengimpor Font 'Poppins' */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* BACKGROUND UTAMA APLIKASI - Abu-abu kebiruan sangat lembut */
    .stApp {
        background-color: #f0f4f8;
    }
    
    /* ==============================================
       STYLING SIDEBAR (FULL COLOR & KONTRAST)
       ============================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A2980 0%, #26D0CE 100%) !important;
        box-shadow: 4px 0px 15px rgba(0,0,0,0.2);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    div[role="radiogroup"] label {
        background-color: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px;
    }

    /* ==============================================
       HEADER BANNER UTAMA
       ============================================== */
    .header-banner {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0px 10px 25px rgba(255, 75, 43, 0.4);
        margin-bottom: 30px;
        margin-top: 10px;
    }
    .header-banner h1 {
        color: white !important;
        font-weight: 900;
        font-size: 3.2rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .header-banner p {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 10px;
        opacity: 0.95;
    }

    /* ==============================================
       STYLING TABS 
       ============================================== */
    div[data-baseweb="tab-list"] {
        background-color: #ffffff;
        border-radius: 50px;
        padding: 5px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        gap: 10px;
        margin-bottom: 20px;
    }
    div[data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 50px !important;
        padding: 10px 30px;
        border: none;
        transition: all 0.3s ease;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%) !important;
        box-shadow: 0px 4px 10px rgba(38, 208, 206, 0.4);
    }
    div[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important;
        font-weight: 700;
    }
    div[data-baseweb="tab"][aria-selected="false"] p {
        color: #5a6a7e !important;
        font-weight: 600;
    }

    /* ==============================================
       KOTAK INPUT & METRIK 
       ============================================== */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 20px !important;
        border-radius: 16px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0; 
        margin-bottom: 15px; 
    }
    .stSlider [data-testid="stMarkdownContainer"] p, 
    .stNumberInput [data-testid="stMarkdownContainer"] p, 
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #1A2980 !important;
        font-weight: 700;
    }

    /* TOMBOL UTAMA */
    button[kind="primary"] {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 15px 30px !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        font-size: 1.1rem !important;
        box-shadow: 0px 8px 20px rgba(255, 75, 43, 0.4) !important;
        transition: 0.3s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0px 12px 25px rgba(255, 75, 43, 0.6) !important;
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
    st.error("❌ File 'model_ensemble_signifikan.pkl' atau 'KEL.2 obesitas Projek MCL 2.xlsx' tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop() 

# ==========================================
# 4. SIDEBAR & MENU NAVIGASI (FULL COLOR)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-weight: 800;'>⚙️ Navigasi</h2>", unsafe_allow_html=True)
    st.write("")
    
    lang = st.radio("Pilih Bahasa / Language", ["Bahasa Indonesia", "English"], label_visibility="collapsed")
    st.write("")
    st.write("")
    
    st.markdown("### 💧 Target Air Harian" if lang == "Bahasa Indonesia" else "### 💧 Daily Water Target")
    lbl_bb = "Berat Badan Anda (kg)" if lang == "Bahasa Indonesia" else "Your Weight (kg)"
    bb_calc = st.number_input(lbl_bb, 30, 200, 60, key="water_calc")
    
    hidrasi_air = bb_calc * 0.033
    lbl_hidrasi = "Kebutuhan Hidrasi Minimum" if lang == "Bahasa Indonesia" else "Minimum Hydration Need"
    lbl_liter = "Liter/hari" if lang == "Bahasa Indonesia" else "Liters/day"
    
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.3); backdrop-filter: blur(5px); margin-top: 10px;">
        <p style="margin:0; font-size: 13px; font-weight: 600;">{lbl_hidrasi}</p>
        <h2 style="margin: 5px 0 0 0; font-weight: 800;">{hidrasi_air:.2f} <span style="font-size:14px; font-weight: normal;">{lbl_liter}</span></h2>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. HEADER BANNER APLIKASI
# ==========================================
st.markdown("""
<div class="header-banner">
    <h1>🩺 Penasihat AI Obesitas</h1>
    <p>⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning ⚡</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. TRANSLASI KELAS UI
# ==========================================
def terjemahkan_hasil_ai(hasil_asli, bahasa):
    kamus_indo = {
        'Insufficient_Weight': 'KEKURANGAN BERAT BADAN',
        'Normal_Weight': 'BERAT BADAN NORMAL',
        'Overweight_Level_I': 'KELEBIHAN BERAT BADAN (Tingkat I)',
        'Overweight_Level_II': 'KELEBIHAN BERAT BADAN (Tingkat II)',
        'Obesity_Type_I': 'OBESITAS (Tipe I)',
        'Obesity_Type_II': 'OBESITAS (Tipe II)',
        'Obesity_Type_III': 'OBESITAS EKSTRIM (Tipe III)'
    }
    if bahasa == "Bahasa Indonesia":
        return kamus_indo.get(hasil_asli, hasil_asli.replace('_', ' ').upper())
    return hasil_asli.replace('_', ' ').upper()

if lang == "Bahasa Indonesia":
    ui = {
        "tabs": ["🎯 Prediksi AI", "💡 Saran Pakar", "📊 Statistik Data"],
        "phys": "👤 Profil Fisik", "gender_lbl": "Jenis Kelamin", "gender_opt": ["Perempuan", "Laki-laki"],
        "age": "Usia (Tahun)", "height": "Tinggi Badan (Meter)", "weight": "Berat Badan (Kilogram)",
        "life": "🍏 Pola Gaya Hidup",
        "fcvc": "Frekuensi Makan Sayur (1: Jarang, 3: Selalu)", "ch2o": "Air Minum (Liter)",
        "faf": "Aktivitas Fisik (0: Pasif, 3: Aktif)", "tue": "Layar Gadget (0: Sebentar, 2: Lama)",
        "btn": "🚀 JALANKAN DIAGNOSIS AI",
        "res_title": "📊 Hasil Analisis Medis", "res_status": "STATUS",
        "lbl_diag": "Hasil Diagnosis", "lbl_bmi": "Nilai BMI", "lbl_conf": "Akurasi AI",
        "tab2_title": "📋 Rekomendasi Medis untuk Status:"
    }
else:
    ui = {
        "tabs": ["🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"],
        "phys": "👤 Physical Profile", "gender_lbl": "Gender", "gender_opt": ["Female", "Male"],
        "age": "Age (Years)", "height": "Height (Meters)", "weight": "Weight (Kilograms)",
        "life": "🍏 Lifestyle Habits",
        "fcvc": "Veggies (1: Rare, 3: Always)", "ch2o": "Water (Liters)",
        "faf": "Exercise (0: Passive, 3: Active)", "tue": "Screen Time (0: Short, 2: Long)",
        "btn": "🚀 RUN AI DIAGNOSTIC",
        "res_title": "📊 Medical Analysis Result", "res_status": "STATUS",
        "lbl_diag": "Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Accuracy",
        "tab2_title": "📋 Medical Recommendations for:"
    }

tab1, tab2, tab3 = st.tabs(ui["tabs"])

# ==========================================
# TAB 1: FORM INPUT UTAMA & PREDIKSI
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='color: #1A2980;'>{ui['phys']}</h3>", unsafe_allow_html=True)
        gender_input = st.selectbox(ui["gender_lbl"], ui["gender_opt"])
        age = st.number_input(ui["age"], 1, 100, 21)
        height = st.number_input(ui["height"], 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input(ui["weight"], 10.0, 250.0, 60.0, step=0.5)

    with col2:
        st.markdown(f"<h3 style='color: #1A2980;'>{ui['life']}</h3>", unsafe_allow_html=True)
        fcvc = st.slider(ui["fcvc"], 1.0, 3.0, 2.0, step=1.0)
        ch2o = st.slider(ui["ch2o"], 1.0, 3.0, 2.0, step=0.5)
        faf = st.slider(ui["faf"], 0.0, 3.0, 1.0, step=1.0)
        tue = st.slider(ui["tue"], 0.0, 2.0, 1.0, step=1.0)

    st.write("")
    
    if st.button(ui["btn"], type="primary", use_container_width=True):
        bmi = weight / (height**2)
        
        gender_to_model = "Female" if gender_input in ["Perempuan", "Female"] else "Male"
        gender_encoded = encoders['Gender'].transform([gender_to_model])[0]
        
        input_data = pd.DataFrame([{
            'Weight': float(weight), 'Height': float(height), 'Age': float(age),
            'FCVC': float(fcvc), 'TUE': float(tue), 'Gender': gender_encoded,
            'FAF': float(faf), 'CH2O': float(ch2o)
        }], columns=feature_names)

        probabilities = model.predict_proba(input_data)[0]
        prediction_idx = np.argmax(probabilities)
        hasil_prediksi_asli = classes[prediction_idx]
        confidence_score = probabilities[prediction_idx] * 100

        hasil_terjemahan = terjemahkan_hasil_ai(hasil_prediksi_asli, lang)

        st.session_state['res_asli'] = hasil_prediksi_asli
        st.session_state['res_terjemahan'] = hasil_terjemahan
        st.session_state['bmi'] = bmi
        st.session_state['conf'] = confidence_score

    # === TAMPILKAN HASIL ===
    if 'res_terjemahan' in st.session_state:
        st.markdown("---")
        st.markdown(f"<h3 style='text-align:center; color: #1A2980;'>{ui['res_title']}</h3>", unsafe_allow_html=True)
        st.write("")

        if "Obesity" in st.session_state['res_asli']:
            st.error(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        elif "Overweight" in st.session_state['res_asli'] or "Insufficient" in st.session_state['res_asli']:
            st.warning(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label=ui["lbl_diag"], value=st.session_state['res_terjemahan'])
        res_col2.metric(label=ui["lbl_bmi"], value=f"{st.session_state['bmi']:.2f}")
        res_col3.metric(label=ui["lbl_conf"], value=f"{st.session_state['conf']:.2f}%")
        
        st.write("")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = st.session_state['bmi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 45], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "rgba(0,0,0,0.3)", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, 18.5], 'color': "#3498db", 'name': 'Kurus'}, 
                    {'range': [18.5, 24.9], 'color': "#2ecc71"}, 
                    {'range': [24.9, 29.9], 'color': "#f1c40f"}, 
                    {'range': [29.9, 45], 'color': "#e74c3c"}],  
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': st.session_state['bmi']}
            }))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        laporan_teks = f"""=====================================
LAPORAN DIAGNOSTIK KESEHATAN AI
=====================================
Status Akhir  : {st.session_state['res_terjemahan']}
Nilai BMI     : {st.session_state['bmi']:.2f}
Akurasi Mesin : {st.session_state['conf']:.2f}%

*Catatan: Dokumen ini dicetak otomatis oleh sistem Penasihat AI Obesitas.
Untuk rekomendasi diet, ikuti arahan dari tenaga medis profesional.
====================================="""
        
        st.download_button(
            label="📥 Unduh Laporan Medis (TXT)",
            data=laporan_teks,
            file_name="Laporan_Kesehatan_AI.txt",
            mime="text/plain",
            type="primary"
        )
        
        st.write("")
        st.info("💡 **Tips:** Buka tab **'Saran Pakar'** di atas untuk panduan gizi lengkap.")

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS & TAUTAN (DIPERBAIKI)
# ==========================================
with tab2:
    if 'res_terjemahan' not in st.session_state:
        st.info("👈 Silakan lakukan prediksi pada tab pertama terlebih dahulu.")
    else:
        st.markdown(f"<h3 style='color: #1A2980;'>{ui['tab2_title']} {st.session_state['res_terjemahan']}</h3>", unsafe_allow_html=True)
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🥗 Panduan Pola Konsumsi")
            if "Obesity" in st.session_state['res_asli']:
                st.error("- Defisit kalori bertahap sangat disarankan.\n- Pangkas karbohidrat berindeks glikemik tinggi.")
            elif "Overweight" in st.session_state['res_asli']:
                st.warning("- Kurangi porsi makanan manis dan gorengan.\n- Perbanyak protein tanpa lemak.")
            else:
                st.success("- Diet Anda seimbang.\n- Pastikan asupan makronutrisi tetap terpenuhi.")
        
        with c2:
            st.markdown("#### 🚴 Panduan Aktivitas Fisik")
            if "Obesity" in st.session_state['res_asli'] or "Overweight" in st.session_state['res_asli']:
                st.error("- Mulai olahraga *low-impact* (berenang/jalan kaki).\n- Targetkan **7000 Langkah/hari**.")
            else:
                st.success("- Kombinasi latihan kardio & beban 3 kali seminggu.\n- Target **10.000 Langkah/hari**.")

        st.write("")
        st.markdown("### 🔗 Referensi Medis Khusus Untuk Anda")
        
        # PERBAIKAN LINK MENGGUNAKAN DIRECT ARTICLE (ANTI ERROR 404)
        if "Insufficient" in st.session_state['res_asli']:
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/cara-menambah-berat-badan" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/expert-answers/underweight/faq-20058429"
            link_text = "📖 Panduan Sehat Menaikkan Berat Badan"
            bg_color = "linear-gradient(135deg, #f6d365, #fda085)" 
        elif "Normal" in st.session_state['res_asli']:
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/pola-hidup-sehat" if lang == "Bahasa Indonesia" else "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
            link_text = "📖 Tips Mempertahankan Gaya Hidup Sehat"
            bg_color = "linear-gradient(135deg, #11998e, #38ef7d)" 
        else: 
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/cara-diet-sehat-dan-cepat" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss/art-20047752"
            link_text = "📖 Panduan Diet Defisit Kalori & Turun BB"
            bg_color = "linear-gradient(135deg, #ff416c, #ff4b2b)" 
            
        st.markdown(f"""
        <a href="{link_url}" target="_blank" style="text-decoration: none;">
            <div style="background: {bg_color}; padding: 15px; border-radius: 12px; text-align: center; color: white; font-weight: 700; font-size: 16px; box-shadow: 0px 10px 20px rgba(0,0,0,0.1); transition: 0.3s;">
                {link_text}
            </div>
        </a>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: VISUALISASI DATASET LATIHAN
# ==========================================
with tab3:
    st.markdown("<h3 style='color: #1A2980;'>📊 Statistik Representatif Dataset</h3>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Obesitas", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Usia", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
