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
# 2. INJEKSI CSS BRUTE-FORCE (TEMA CERAH, ELEGAN & PROFESIONAL)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

    .stApp { background-color: #f8fafc !important; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e0f2fe 0%, #bae6fd 100%) !important;
        box-shadow: 4px 0px 15px rgba(0,0,0,0.05) !important;
        border-right: 2px solid #7dd3fc !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    
    .header-banner {
        background: linear-gradient(135deg, #1A2980 0%, #26D0CE 100%);
        padding: 40px 20px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 10px 20px rgba(38, 208, 206, 0.25);
        margin: 10px 0px 30px 0px;
    }
    .header-banner h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 3.2rem !important;
        margin: 0;
        letter-spacing: -1px;
    }
    .header-banner p {
        color: #e0f2fe !important;
        font-size: 1.15rem;
        font-weight: 500;
        margin-top: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #f1f5f9 !important; 
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
        transition: 0.3s;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%) !important;
        border: 2px solid #1A2980 !important;
        box-shadow: 0px 6px 15px rgba(38, 208, 206, 0.4) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 20px !important;
        border-radius: 15px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.03) !important;
    }
    .stSlider [data-testid="stMarkdownContainer"] p, 
    .stNumberInput [data-testid="stMarkdownContainer"] p, 
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #1A2980 !important;
        font-weight: 700 !important;
    }

    button[kind="primary"] {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 15px 30px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        box-shadow: 0px 8px 20px rgba(38, 208, 206, 0.4) !important;
        transition: 0.3s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0px 12px 25px rgba(38, 208, 206, 0.6) !important;
    }
    
    .welcome-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 8vh;
    }
    .welcome-card {
        background: #ffffff;
        padding: 50px 40px;
        border-radius: 20px;
        box-shadow: 0px 20px 40px rgba(0,0,0,0.08);
        border-top: 6px solid #26D0CE;
        text-align: center;
        max-width: 800px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. TAMPILAN AWAL (WELCOME SCREEN)
# ==========================================
if 'welcomed' not in st.session_state:
    st.session_state['welcomed'] = False

if not st.session_state['welcomed']:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-card">
            <h1 style="font-size: 3rem; color: #1A2980; font-weight: 900; margin-bottom: 10px;">👋 Selamat Datang!</h1>
            <h3 style="color: #26D0CE; font-weight: 700; margin-top: 0;">di Penasihat AI Obesitas</h3>
            <p style="color: #475569; font-size: 1.15rem; line-height: 1.6; margin-top: 20px; margin-bottom: 40px;">
                Ini adalah portal diagnostik kesehatan pintar Anda. Kami menggunakan teknologi <b>Ensemble Machine Learning</b> canggih untuk memprediksi tingkat risiko obesitas, menganalisis gaya hidup, dan memberikan saran pakar secara instan dan akurat.
            </p>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🚀 Mulai Analisis Kesehatan Anda Sekarang", type="primary", use_container_width=True):
            st.session_state['welcomed'] = True
            st.rerun()
    st.stop() 

# ==========================================
# 4. FUNGSI LOADING DATA & CACHING
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
# 5. SIDEBAR MENU
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
    st.markdown("<h2 style='color: #1A2980; font-weight: 900; margin-top: -10px;'>Menu Utama</h2>", unsafe_allow_html=True)
    
    lang = st.radio("Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.markdown("### 💧 Pelacak Hidrasi Harian" if lang == "Bahasa Indonesia" else "### 💧 Daily Water Tracker")
    lbl_bb = "Berat Badan Anda (kg)" if lang == "Bahasa Indonesia" else "Your Weight (kg)"
    bb_calc = st.number_input(lbl_bb, 30, 200, 60, key="water_calc")
    
    hidrasi_air = bb_calc * 0.033 
    
    st.markdown(f"""
    <div style="background: #ffffff; padding: 15px; border-radius: 12px; border: 2px solid #0ea5e9; box-shadow: 0px 4px 10px rgba(14, 165, 233, 0.2); text-align: center; margin-bottom: 20px;">
        <p style="margin:0; font-size: 13px; color:#475569;">{'Target Minimum Anda:' if lang == 'Bahasa Indonesia' else 'Your Minimum Target:'}</p>
        <h2 style="margin: 5px 0 0 0; font-weight: 800; color:#0284c7;">{hidrasi_air:.2f} <span style="font-size:14px;">{'Liter/hari' if lang == 'Bahasa Indonesia' else 'Liters/day'}</span></h2>
    </div>
    """, unsafe_allow_html=True)

    lbl_minum = "Sudah minum berapa banyak hari ini? (Liter)" if lang == "Bahasa Indonesia" else "How much have you drank today? (Liters)"
    air_diminum = st.slider(lbl_minum, 0.0, 5.0, 0.0, step=0.1)
    
    persentase_hidrasi = min(air_diminum / hidrasi_air, 1.0)
    st.progress(persentase_hidrasi)
    
    if air_diminum >= hidrasi_air:
        st.success("🎉 Target hidrasi harian tercapai!" if lang == "Bahasa Indonesia" else "🎉 Daily hydration target met!")

# ==========================================
# 6. HEADER BANNER APLIKASI
# ==========================================
st.markdown("""
<div class="header-banner">
    <h1>🩺 Penasihat AI Obesitas</h1>
    <p>⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning ⚡</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. TRANSLASI KELAS UI & KAMUS BAHASA
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
        "fcvc": "Frekuensi Makan Sayur (1: Jarang, 3: Selalu)", "ch2o": "Air Minum Harian (Liter)",
        "faf": "Aktivitas Fisik (0: Pasif, 3: Aktif)", "tue": "Layar Gadget (0: Sebentar, 2: Lama)",
        "btn": "🚀 JALANKAN DIAGNOSIS AI",
        "res_title": "📊 Hasil Analisis Medis", "res_status": "STATUS",
        "lbl_diag": "Hasil Diagnosis", "lbl_bmi": "Nilai BMI", "lbl_conf": "Akurasi AI",
        "tab2_title": "📋 Rekomendasi Medis untuk Status:",
        "alert_saran": "💡 <b>Langkah Selanjutnya:</b> Klik tab <b>Saran Pakar</b> di bagian atas untuk melihat panduan gizi dan rekomendasi kesehatan khusus untuk Anda!"
    }
else:
    ui = {
        "tabs": ["🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"],
        "phys": "👤 Physical Profile", "gender_lbl": "Gender", "gender_opt": ["Female", "Male"],
        "age": "Age (Years)", "height": "Height (Meters)", "weight": "Weight (Kilograms)",
        "life": "🍏 Lifestyle Habits",
        "fcvc": "Veggies (1: Rare, 3: Always)", "ch2o": "Daily Water (Liters)",
        "faf": "Exercise (0: Passive, 3: Active)", "tue": "Screen Time (0: Short, 2: Long)",
        "btn": "🚀 RUN AI DIAGNOSTIC",
        "res_title": "📊 Medical Analysis Result", "res_status": "STATUS",
        "lbl_diag": "Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Accuracy",
        "tab2_title": "📋 Medical Recommendations for:",
        "alert_saran": "💡 <b>Next Step:</b> Click the <b>Expert Advice</b> tab above to view custom nutrition guides and health recommendations!"
    }

tab1, tab2, tab3 = st.tabs(ui["tabs"])

# ==========================================
# TAB 1: FORM INPUT UTAMA & PREDIKSI
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['phys']}</h3>", unsafe_allow_html=True)
        gender_input = st.selectbox(ui["gender_lbl"], ui["gender_opt"])
        age = st.number_input(ui["age"], 1, 100, 21)
        height = st.number_input(ui["height"], 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input(ui["weight"], 10.0, 250.0, 60.0, step=0.5)

    with col2:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['life']}</h3>", unsafe_allow_html=True)
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

    # === TAMPILKAN HASIL JIKA SUDAH ADA DI SESSION ===
    if 'res_terjemahan' in st.session_state:
        st.markdown("---")
        st.markdown(f"<div style='background:#ffffff; padding:15px; border-radius:15px; border: 2px solid #26D0CE; text-align:center; box-shadow: 0px 5px 15px rgba(0,0,0,0.05);'><h2 style='color: #1A2980; font-weight:900; margin:0;'>{ui['res_title']}</h2></div>", unsafe_allow_html=True)
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
                'axis': {'range': [None, 45], 'tickwidth': 2, 'tickcolor': "black"},
                'bar': {'color': "rgba(0,0,0,0.4)", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 18.5], 'color': "#3498db", 'name': 'Kurus'}, 
                    {'range': [18.5, 24.9], 'color': "#2ecc71"}, 
                    {'range': [24.9, 29.9], 'color': "#f1c40f"}, 
                    {'range': [29.9, 45], 'color': "#e74c3c"}],  
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 0.85,
                    'value': st.session_state['bmi']}
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); padding: 18px 25px; border-radius: 15px; margin-top: 20px; margin-bottom: 25px; box-shadow: 0px 8px 20px rgba(255, 75, 43, 0.4); text-align: center; border: 2px solid white;">
            <p style="color: white !important; font-size: 1.1rem; margin: 0; font-weight: 500;">{ui['alert_saran']}</p>
        </div>
        """, unsafe_allow_html=True)

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

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS & TAUTAN (DENGAN 7 KATEGORI)
# ==========================================
with tab2:
    if 'res_terjemahan' not in st.session_state:
        st.info("👈 Silakan lakukan prediksi pada tab pertama terlebih dahulu.")
    else:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['tab2_title']} {st.session_state['res_terjemahan']}</h3>", unsafe_allow_html=True)
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
        
        # --- LOGIKA 7 KATEGORI LINK MEDIS ---
        kategori = st.session_state['res_asli']
        
        # 1. Kurus (Insufficient Weight)
        if kategori == 'Insufficient_Weight':
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/cara-menambah-berat-badan" if lang == "Bahasa Indonesia" else "https://www.healthline.com/nutrition/how-to-gain-weight"
            link_text = "📖 Cara Sehat Menambah Berat Badan & Massa Otot" if lang == "Bahasa Indonesia" else "📖 Healthy Ways to Gain Weight"
            bg_color = "linear-gradient(135deg, #f6d365, #fda085)" 
            
        # 2. Normal (Normal Weight)
        elif kategori == 'Normal_Weight':
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/pola-hidup-sehat" if lang == "Bahasa Indonesia" else "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
            link_text = "📖 Panduan Mempertahankan Pola Hidup Sehat" if lang == "Bahasa Indonesia" else "📖 Guide to Maintaining a Healthy Diet"
            bg_color = "linear-gradient(135deg, #11998e, #38ef7d)" 
            
        # 3. Overweight Tingkat 1
        elif kategori == 'Overweight_Level_I':
            link_url = "https://www.halodoc.com/artikel/ini-cara-diet-sehat-untuk-menurunkan-berat-badan" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss/art-20047752"
            link_text = "📖 Panduan Defisit Kalori Pemula (Turun BB Ringan)" if lang == "Bahasa Indonesia" else "📖 Basic Calorie Deficit & Weight Loss Guide"
            bg_color = "linear-gradient(135deg, #f2994a, #f2c94c)" 

        # 4. Overweight Tingkat 2
        elif kategori == 'Overweight_Level_II':
            link_url = "https://www.alodokter.com/diet-sehat-untuk-menurunkan-berat-badan" if lang == "Bahasa Indonesia" else "https://www.healthline.com/nutrition/how-to-lose-weight-as-fast-as-possible"
            link_text = "📖 Strategi Efektif Penurunan Berat Badan (Diet Sehat)" if lang == "Bahasa Indonesia" else "📖 Effective Weight Loss Strategies"
            bg_color = "linear-gradient(135deg, #FF416C, #FF4B2B)" 

        # 5. Obesitas Tipe 1
        elif kategori == 'Obesity_Type_I':
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/apa-itu-obesitas" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/diseases-conditions/obesity/diagnosis-treatment/drc-20375749"
            link_text = "📖 Pemahaman Bahaya Obesitas & Penanganan Dasar" if lang == "Bahasa Indonesia" else "📖 Obesity Management and Diagnosis"
            bg_color = "linear-gradient(135deg, #e52d27, #b31217)" 

        # 6. Obesitas Tipe 2
        elif kategori == 'Obesity_Type_II':
            link_url = "https://www.alodokter.com/obesitas" if lang == "Bahasa Indonesia" else "https://www.clevelandclinic.org/health/diseases/11209-weight-control-and-obesity"
            link_text = "📖 Penanganan Medis Obesitas Lanjut & Olahraga Khusus" if lang == "Bahasa Indonesia" else "📖 Advanced Weight Control and Obesity Care"
            bg_color = "linear-gradient(135deg, #cb2d3e, #ef473a)" 

        # 7. Obesitas Tipe 3 (Ekstrim)
        else: 
            link_url = "https://www.halodoc.com/artikel/obesitas-morbid-ketahui-penyebab-dan-cara-mengatasinya" if lang == "Bahasa Indonesia" else "https://www.healthline.com/health/morbid-obesity"
            link_text = "📖 Panduan Penanganan Medis Obesitas Ekstrim/Morbid" if lang == "Bahasa Indonesia" else "📖 Guide to Medical Intervention for Morbid Obesity"
            bg_color = "linear-gradient(135deg, #870000, #190a05)" 
            
        st.markdown(f"""
        <a href="{link_url}" target="_blank" style="text-decoration: none;">
            <div style="background: {bg_color}; padding: 15px; border-radius: 12px; text-align: center; color: white; font-weight: 800; font-size: 16px; box-shadow: 0px 5px 15px rgba(0,0,0,0.15); border: 2px solid white;">
                {link_text}
            </div>
        </a>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: VISUALISASI DATASET LATIHAN
# ==========================================
with tab3:
    st.markdown("<h3 style='color: #1A2980; font-weight:800;'>📊 Statistik Representatif Dataset</h3>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Obesitas", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Usia", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
