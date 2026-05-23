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
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CUSTOM CSS (GAYA PREMIUM & ELEGAN)
# ==========================================
st.markdown("""
<style>
    /* Font Poppins yang elegan dan modern */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* BACKGROUND UTAMA - Sangat bersih (Off-White) */
    .stApp {
        background-color: #f4f7f6;
    }
    
    /* ==============================================
       STYLING UNTUK WELCOME SCREEN (LAYAR SAMBUTAN)
       ============================================== */
    .welcome-card {
        background-color: #ffffff;
        padding: 50px 40px;
        border-radius: 24px;
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid #eaedf1;
        margin-top: 20px;
        transition: all 0.3s ease;
    }
    .welcome-card:hover {
        transform: translateY(-5px);
        box-shadow: 0px 30px 50px rgba(0, 0, 0, 0.12);
    }
    .icon-logo {
        font-size: 55px;
        margin-bottom: 10px;
    }
    .welcome-title {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 15px;
        letter-spacing: -1px;
    }
    .welcome-subtitle {
        color: #5a6a7e;
        font-size: 1.15rem;
        line-height: 1.6;
        font-weight: 400;
        margin-bottom: 35px;
        padding: 0 20px;
    }

    /* ==============================================
       STYLING KOMPONEN APLIKASI
       ============================================== */
    /* Desain Tab (Elegan & Halus) */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    div[data-baseweb="tab"] {
        background-color: #e2e8f0; 
        border-radius: 12px 12px 0px 0px;
        padding: 10px 24px;
        border: none;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    div[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important;
        font-weight: 700;
        font-size: 1.05rem;
    }

    /* Sidebar Bersih */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 5px 0px 20px rgba(0, 0, 0, 0.04); 
        border-right: none;
    }

    /* Kotak Input / Form (Sleek) */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 15px 20px !important;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0; 
        margin-bottom: 12px; 
    }

    /* Tombol Utama (Premium Dark Blue) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 14px 30px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        font-size: 1.05rem !important;
        box-shadow: 0px 10px 20px rgba(15, 32, 39, 0.3) !important;
        transition: 0.4s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0px 15px 25px rgba(15, 32, 39, 0.4) !important;
        background: linear-gradient(135deg, #142b35 0%, #294a56 50%, #3a6d84 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LAYAR SAMBUTAN (WELCOME SCREEN ELEGAN)
# ==========================================
if 'welcomed' not in st.session_state:
    st.session_state['welcomed'] = False

if not st.session_state['welcomed']:
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
    colA, colB, colC = st.columns([1, 2.5, 1])
    with colB:
        # Dibungkus dalam div .welcome-card agar melayang dan elegan
        st.markdown("""
        <div class='welcome-card'>
            <div class='icon-logo'>🤖⚕️</div>
            <h1 class='welcome-title'>Penasihat AI Obesitas</h1>
            <p class='welcome-subtitle'>Sistem diagnostik kesehatan digital masa depan. Kami menggunakan kekuatan <b>Ensemble Machine Learning</b> untuk menganalisis metrik tubuh Anda dengan presisi tingkat medis yang tinggi.</p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Masuk ke Ruang Analisis", type="primary", use_container_width=True):
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
    st.error("❌ File 'model_ensemble_signifikan.pkl' atau 'KEL.2 obesitas Projek MCL 2.xlsx' tidak ditemukan! Pastikan file berada di folder yang sama.")
    st.stop() 

# ==========================================
# 5. HEADER APLIKASI (SETELAH MASUK)
# ==========================================
# Header setelah masuk web dibikin sedikit lebih minimalis
st.markdown("<h2 style='text-align: center; color: #1e3c72; font-weight: 800;'>🤖 Penasihat AI Obesitas</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5a6a7e; margin-bottom: 30px;'>Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning</p>", unsafe_allow_html=True)

# ==========================================
# 6. SIDEBAR & MENU NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=90)
    st.title("Menu / Nav")
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.subheader("💧 Target Air Harian" if lang == "Bahasa Indonesia" else "💧 Daily Water Target")
    lbl_bb = "Berat Badan Anda (kg)" if lang == "Bahasa Indonesia" else "Your Weight (kg)"
    bb_calc = st.number_input(lbl_bb, 30, 200, 60, key="water_calc")
    
    hidrasi_air = bb_calc * 0.033
    lbl_hidrasi = "Kebutuhan Hidrasi Minimum" if lang == "Bahasa Indonesia" else "Minimum Hydration Need"
    lbl_liter = "Liter/hari" if lang == "Bahasa Indonesia" else "Liters/day"
    
    st.markdown(f"""
    <div style="background: #ffffff; padding: 15px; border-radius: 12px; border-left: 5px solid #2a5298; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);">
        <p style="margin:0; color: #5a6a7e; font-size: 13px; font-weight: 600;">{lbl_hidrasi}</p>
        <h2 style="margin: 5px 0 0 0; color: #1e3c72;">{hidrasi_air:.2f} <span style="font-size:14px; font-weight: normal;">{lbl_liter}</span></h2>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 7. TRANSLASI KELAS UI
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
        "btn": "✨ PROSES DIAGNOSIS",
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
        "btn": "✨ PROCESS DIAGNOSIS",
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
        st.subheader(ui["phys"])
        gender_input = st.selectbox(ui["gender_lbl"], ui["gender_opt"])
        age = st.number_input(ui["age"], 1, 100, 21)
        height = st.number_input(ui["height"], 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input(ui["weight"], 10.0, 250.0, 60.0, step=0.5)

    with col2:
        st.subheader(ui["life"])
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
        st.subheader(ui["res_title"])

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
        st.markdown("### 🎛️ Indikator Indeks Massa Tubuh (BMI)")
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
        st.info("💡 **Tips:** Untuk panduan gizi lengkap dan *link* medis, silakan buka tab **'Saran Pakar'** di atas.")

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS & TAUTAN EKSTERNAL
# ==========================================
with tab2:
    if 'res_terjemahan' not in st.session_state:
        st.info("👈 Silakan lakukan prediksi pada tab pertama terlebih dahulu.")
    else:
        st.subheader(f"{ui['tab2_title']} {st.session_state['res_terjemahan']}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🥗 Panduan Pola Konsumsi")
            if "Obesity" in st.session_state['res_asli']:
                st.error("- Defisit kalori bertahap sangat disarankan.\n- Pangkas karbohidrat berindeks glikemik tinggi.")
            elif "Overweight" in st.session_state['res_asli']:
                st.warning("- Kurangi porsi makanan manis dan gorengan.\n- Perbanyak protein tanpa lemak.")
            else:
                st.success("- Diet Anda seimbang.\n- Pastikan asupan makronutrisi tetap terpenuhi.")
        
        with c2:
            st.subheader("🚴 Panduan Aktivitas Fisik")
            if "Obesity" in st.session_state['res_asli'] or "Overweight" in st.session_state['res_asli']:
                st.error("- Mulai olahraga *low-impact* (berenang/jalan kaki).\n- Targetkan **7000 Langkah/hari**.")
            else:
                st.success("- Kombinasi latihan kardio & beban 3 kali seminggu.\n- Target **10.000 Langkah/hari**.")

        st.write("")
        st.markdown("### 🔗 Referensi Medis Khusus Untuk Anda")
        
        if "Insufficient" in st.session_state['res_asli']:
            link_url = "https://www.alodokter.com/search?q=cara+menaikkan+berat+badan" if lang == "Bahasa Indonesia" else "https://www.healthline.com/search?q1=gain+weight"
            link_text = "📖 Panduan Sehat Menaikkan Berat Badan"
            bg_color = "linear-gradient(135deg, #f6d365, #fda085)" 
        elif "Normal" in st.session_state['res_asli']:
            link_url = "https://www.halodoc.com/artikel/kategori/kesehatan" if lang == "Bahasa Indonesia" else "https://www.who.int/initiatives/behealthy"
            link_text = "📖 Tips Mempertahankan Gaya Hidup Sehat"
            bg_color = "linear-gradient(135deg, #11998e, #38ef7d)" 
        else: 
            link_url = "https://www.alodokter.com/search?q=diet+defisit+kalori" if lang == "Bahasa Indonesia" else "https://www.healthline.com/search?q1=weight+loss+diet"
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
    st.subheader("📊 Statistik Representatif Dataset")
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Obesitas", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Usia", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
