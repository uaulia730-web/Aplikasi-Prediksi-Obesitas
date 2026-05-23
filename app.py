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
    page_icon="🥗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CUSTOM CSS (DESAIN ELEGAN, FONT CANTIK & TAB BERWARNA)
# ==========================================
st.markdown("""
<style>
    /* Mengimpor Font 'Poppins' dari Google Fonts yang sangat cantik & modern */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800;900&display=swap');
    
    /* Mengubah seluruh font di aplikasi */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }

    /* BACKGROUND UTAMA - Solid, Bersih, Elegan (Off-White/Pearl) */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* JUDUL APLIKASI (TIDAK BASIC) - Efek teks gradien dengan bayangan */
    .judul-utama {
        text-align: center;
        background: linear-gradient(45deg, #FF512F, #DD2476, #FF512F);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 900;
        margin-bottom: 0px;
        padding-bottom: 10px;
        filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.1));
        animation: textShine 4s ease-in-out infinite;
    }
    @keyframes textShine {
        to { background-position: 200% center; }
    }

    /* SUB-JUDUL */
    .sub-judul {
        text-align: center;
        color: #7f8c8d;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* DESAIN TAB MENU (AGAR TIDAK POLOS) */
    div[data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    div[data-baseweb="tab"] {
        background-color: #e9ecef; /* Warna tab saat tidak aktif */
        border-radius: 12px 12px 0px 0px;
        padding: 10px 24px;
        border: none;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #FF512F; /* Warna mencolok saat tab dipilih */
    }
    div[data-baseweb="tab"][aria-selected="true"] p {
        color: white !important;
        font-weight: 800;
        font-size: 1.1rem;
    }

    /* SIDEBAR KONTRAST */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 4px 0px 15px rgba(0, 0, 0, 0.05); 
        border-right: none;
    }

    /* KOTAK INPUT (SOLID WHITE & SHADOW) */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 15px 20px !important;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.04);
        border: 1px solid #e0e0e0; 
        margin-bottom: 12px; 
    }

    /* TOMBOL UTAMA */
    button[kind="primary"] {
        background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 12px 30px !important;
        font-weight: 800 !important;
        letter-spacing: 1px;
        box-shadow: 0px 8px 15px rgba(31, 162, 255, 0.3) !important;
        transition: 0.3s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0px 12px 20px rgba(31, 162, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LAYAR SAMBUTAN (WELCOME SCREEN)
# ==========================================
if 'welcomed' not in st.session_state:
    st.session_state['welcomed'] = False

if not st.session_state['welcomed']:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    colA, colB, colC = st.columns([1, 2, 1])
    with colB:
        st.markdown("<h1 class='judul-utama'>🥗 Penasihat AI Obesitas</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-judul'>Selamat datang di masa depan diagnostik kesehatan digital. Kami menggunakan Ensemble Machine Learning untuk menganalisis metrik tubuh Anda dengan tingkat akurasi tinggi.</p>", unsafe_allow_html=True)
        st.write("")
        if st.button("🚀 Mulai Analisis Kesehatan Anda", type="primary", use_container_width=True):
            st.session_state['welcomed'] = True
            st.rerun() # Refresh halaman secara instan untuk masuk ke web
    st.stop() # Hentikan eksekusi kode di bawahnya sampai tombol ditekan

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
st.markdown("<h1 class='judul-utama'>🥗 Penasihat AI Obesitas</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-judul'>⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning</p>", unsafe_allow_html=True)
st.divider()

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
    <div style="background: #ffffff; padding: 15px; border-radius: 12px; border-left: 5px solid #FF512F; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);">
        <p style="margin:0; color: #7f8c8d; font-size: 13px; font-weight: bold;">{lbl_hidrasi}</p>
        <h2 style="margin: 5px 0 0 0; color: #2c3e50;">{hidrasi_air:.2f} <span style="font-size:14px; font-weight: normal;">{lbl_liter}</span></h2>
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
        "btn": "🚀 HITUNG DIAGNOSIS SEKARANG",
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
        "btn": "🚀 RUN DIAGNOSTIC NOW",
        "res_title": "📊 Medical Analysis Result", "res_status": "STATUS",
        "lbl_diag": "Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Accuracy",
        "tab2_title": "📋 Medical Recommendations for:"
    }

# MEMBUAT TABS (Desain Tab sudah diatur di CSS atas)
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

    # === TAMPILKAN HASIL JIKA SUDAH ADA DI SESSION ===
    if 'res_terjemahan' in st.session_state:
        st.markdown("---")
        st.subheader(ui["res_title"])

        if "Obesity" in st.session_state['res_asli']:
            st.error(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        elif "Overweight" in st.session_state['res_asli'] or "Insufficient" in st.session_state['res_asli']:
            st.warning(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")

        # 3 Metrik
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label=ui["lbl_diag"], value=st.session_state['res_terjemahan'])
        res_col2.metric(label=ui["lbl_bmi"], value=f"{st.session_state['bmi']:.2f}")
        res_col3.metric(label=ui["lbl_conf"], value=f"{st.session_state['conf']:.2f}%")
        
        # --- FITUR BARU 1: VISUALISASI METERAN BMI (GAUGE CHART) ---
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
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 18.5], 'color': "#3498db", 'name': 'Kurus'}, # Biru
                    {'range': [18.5, 24.9], 'color': "#2ecc71"}, # Hijau (Normal)
                    {'range': [24.9, 29.9], 'color': "#f1c40f"}, # Kuning (Overweight)
                    {'range': [29.9, 45], 'color': "#e74c3c"}],  # Merah (Obesitas)
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': st.session_state['bmi']}
            }))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- FITUR BARU 2: CETAK LAPORAN (DOWNLOAD BUTTON) ---
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
            label="📥 Cetak & Unduh Laporan Medis (TXT)",
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

        # --- PERBAIKAN LINK (SMART SEARCH URLS - ANTI 404) ---
        st.write("")
        st.markdown("### 🔗 Referensi Medis Khusus Untuk Anda")
        
        if "Insufficient" in st.session_state['res_asli']:
            link_url = "https://www.alodokter.com/search?q=cara+menaikkan+berat+badan" if lang == "Bahasa Indonesia" else "https://www.healthline.com/search?q1=gain+weight"
            link_text = "📖 Panduan Sehat Menaikkan Berat Badan"
            bg_color = "linear-gradient(to right, #f6d365, #fda085)" 
        elif "Normal" in st.session_state['res_asli']:
            link_url = "https://www.halodoc.com/artikel/kategori/kesehatan" if lang == "Bahasa Indonesia" else "https://www.who.int/initiatives/behealthy"
            link_text = "📖 Tips Mempertahankan Gaya Hidup Sehat"
            bg_color = "linear-gradient(to right, #84fab0, #8fd3f4)" 
        else: 
            link_url = "https://www.alodokter.com/search?q=diet+defisit+kalori" if lang == "Bahasa Indonesia" else "https://www.healthline.com/search?q1=weight+loss+diet"
            link_text = "📖 Panduan Diet Defisit Kalori & Turun BB"
            bg_color = "linear-gradient(to right, #ff758c, #ff7eb3)" 
            
        st.markdown(f"""
        <a href="{link_url}" target="_blank" style="text-decoration: none;">
            <div style="background: {bg_color}; padding: 15px; border-radius: 8px; text-align: center; color: white; font-weight: 800; font-size: 16px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);">
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
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Obesitas", hole=0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Usia", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
