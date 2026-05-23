import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import time

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Obesitas - Ensemble Learning", 
    page_icon="🥗", 
    layout="wide" 
)

# ==========================================
# 2. INJEKSI CUSTOM CSS (ANIMASI GRADASI & HIGH CONTRAST)
# ==========================================
st.markdown("""
<style>
    /* ANIMASI GRADASI BERGERAK UNTUK BACKGROUND UTAMA */
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background: linear-gradient(-45deg, #e0c3fc, #8ec5fc, #a8edea, #fed6e3);
        background-size: 400% 400%;
        animation: gradientAnimation 12s ease infinite; 
    }
    
    /* MEMPERCANTIK SIDEBAR DENGAN KONTRAST TINGGI */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 4px 0px 15px rgba(0, 0, 0, 0.05); 
        border-right: none;
    }

    /* KONTRAST TEKS HEADER */
    h1, h2, h3 {
        color: #1c2833 !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 800 !important;
        text-shadow: 1px 1px 0px rgba(255,255,255,0.8);
    }

    /* BATAS ANTARA TOOLS (GLASSMORPHISM EFFECT) */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.85); 
        padding: 15px 20px !important;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin-bottom: 12px; 
    }

    /* KONTRAST TEKS LABEL SLIDER & INPUT */
    .stSlider [data-testid="stMarkdownContainer"] p, 
    .stNumberInput [data-testid="stMarkdownContainer"] p, 
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
        font-weight: 700;
        font-size: 15px;
    }

    /* MEMPERCANTIK TOMBOL UTAMA (ANALISIS) DENGAN GRADASI */
    button[kind="primary"] {
        background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 51%, #1fa2ff 100%) !important;
        background-size: 200% auto !important;
        color: white !important;
        border-radius: 25px !important; 
        border: none !important;
        padding: 12px 30px !important;
        font-weight: 800 !important;
        letter-spacing: 1.5px;
        box-shadow: 0px 8px 15px rgba(31, 162, 255, 0.4) !important;
        transition: 0.5s !important;
    }
    button[kind="primary"]:hover {
        background-position: right center !important; 
        box-shadow: 0px 12px 20px rgba(31, 162, 255, 0.6) !important;
        transform: translateY(-3px);
    }

    /* KOTAK METRIK HASIL (CARD 3D) */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
        border-left: 8px solid #1fa2ff; 
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

if 'initialized' not in st.session_state:
    with st.spinner("🔮 Memuat Model Ensemble Learning..."):
        time.sleep(1.0)
    st.toast("✨ Sistem AI Siap Digunakan!", icon="🚀")
    st.session_state['initialized'] = True

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
# 4. SIDEBAR & MENU NAVIGASI
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
    <div style="background: #f8f9fa; padding: 15px; border-radius: 12px; border: 2px solid #3498db; box-shadow: 0px 4px 6px rgba(0,0,0,0.05);">
        <p style="margin:0; color: #7f8c8d; font-size: 13px; font-weight: bold;">{lbl_hidrasi}</p>
        <h2 style="margin: 5px 0 0 0; color: #2980b9;">{hidrasi_air:.2f} <span style="font-size:14px; font-weight: normal;">{lbl_liter}</span></h2>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. TRANSLASI & KAMUS UI
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
        "title": "🥗 Penasihat AI Obesitas",
        "subtitle": "⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning",
        "tabs": ["🎯 Prediksi AI", "💡 Saran Pakar", "📊 Statistik Data"],
        "phys": "👤 Profil Fisik", "gender_lbl": "Jenis Kelamin", "gender_opt": ["Perempuan", "Laki-laki"],
        "age": "Usia (Tahun)", "height": "Tinggi Badan (Meter)", "weight": "Berat Badan (Kilogram)",
        "life": "🍏 Pola Gaya Hidup",
        "fcvc": "Frekuensi Makan Sayur (1: Jarang, 2: Kadang, 3: Selalu)",
        "ch2o": "Konsumsi Air Minum Harian (Liter)",
        "faf": "Aktivitas Fisik / Olahraga (0: Pasif, 3: Sangat Aktif)",
        "tue": "Waktu Layar Gadget (0: Sebentar, 2: Lama Semalaman)",
        "btn": "🚀 MENGHITUNG PREDIKSI",
        "load1": "🧠 Model sedang memproses data tabular...",
        "load2": "🎯 Menghitung probabilitas kelas...",
        "load3": "✅ Inferensi Selesai!",
        "res_title": "📊 Hasil Analisis Medis",
        "res_status": "STATUS KESEHATAN",
        "lbl_diag": "Hasil Diagnosis Akhir", "lbl_bmi": "Nilai BMI", "lbl_conf": "Tingkat Keyakinan AI",
        "tab2_warn": "👈 Silakan jalankan analisis pada tab pertama terlebih dahulu.",
        "tab2_title": "📋 Rekomendasi Medis untuk Status:",
        "food_title": "🥗 Panduan Pola Konsumsi",
        "sport_title": "🚴 Panduan Aktivitas Fisik",
        "chart_title": "📊 Statistik Representatif Grafik Dataset",
        "goto_expert": "💡 **Untuk panduan gizi dan aktivitas fisik, silakan buka tab 'Saran Pakar' di atas.**"
    }
else:
    ui = {
        "title": "🥗 Obesity AI Advisor",
        "subtitle": "⚡ Instant Health Diagnostic Powered by Ensemble Learning",
        "tabs": ["🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"],
        "phys": "👤 Physical Profile", "gender_lbl": "Gender", "gender_opt": ["Female", "Male"],
        "age": "Age (Years)", "height": "Height (Meters)", "weight": "Weight (Kilograms)",
        "life": "🍏 Lifestyle Habits",
        "fcvc": "Vegetable Consumption Frequency (1: Rare, 3: Always)",
        "ch2o": "Daily Water Intake (Liters)",
        "faf": "Physical Activity / Exercise (0: Passive, 3: Very Active)",
        "tue": "Screen Time (0: Short, 2: Long/Overnight)",
        "btn": "🚀 CALCULATE PREDICTION",
        "load1": "🧠 Model is processing tabular data...",
        "load2": "🎯 Calculating class probabilities...",
        "load3": "✅ Inference Complete!",
        "res_title": "📊 Medical Analysis Result",
        "res_status": "HEALTH STATUS",
        "lbl_diag": "Final Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Confidence Level",
        "tab2_warn": "👈 Please run the analysis on the first tab first.",
        "tab2_title": "📋 Medical Recommendations for:",
        "food_title": "🥗 Diet & Nutrition Guide",
        "sport_title": "🚴 Physical Activity Guide",
        "chart_title": "📊 Dataset Statistics Representation",
        "goto_expert": "💡 **For nutrition guides and physical activity advice, please open the 'Expert Advice' tab above.**"
    }

st.title(ui["title"])
st.markdown(f"**{ui['subtitle']}**")
st.divider()

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
        with st.status(ui["load1"], expanded=True) as status:
            time.sleep(0.6)
            status.update(label=ui["load2"], state="running")
            time.sleep(0.4)
            status.update(label=ui["load3"], state="complete")
            
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

        st.markdown("---")
        st.subheader(ui["res_title"])

        if "Obesity" in hasil_prediksi_asli:
            st.error(f"⚠️ **{ui['res_status']}: {hasil_terjemahan}**")
        elif "Overweight" in hasil_prediksi_asli or "Insufficient" in hasil_prediksi_asli:
            st.warning(f"⚠️ **{ui['res_status']}: {hasil_terjemahan}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {hasil_terjemahan}**")
            st.balloons()

        st.write("")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label=ui["lbl_diag"], value=hasil_terjemahan)
        with res_col2:
            st.metric(label=ui["lbl_bmi"], value=f"{bmi:.2f}")
        with res_col3:
            st.metric(label=ui["lbl_conf"], value=f"{confidence_score:.2f}%")
        
        st.write("")
        st.progress(int(confidence_score) / 100)
        
        # --- PEMBERITAHUAN UNTUK MELIHAT TAB SARAN PAKAR DITAMBAHKAN DI SINI ---
        st.write("")
        st.info(ui["goto_expert"])

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS
# ==========================================
with tab2:
    if 'res_terjemahan' not in st.session_state:
        st.info(ui["tab2_warn"])
    else:
        st.subheader(f"{ui['tab2_title']} {st.session_state['res_terjemahan']}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(ui["food_title"])
            if "Obesity" in st.session_state['res_asli']:
                if lang == "Bahasa Indonesia":
                    st.error("- Pangkas konsumsi karbohidrat berindeks glikemik tinggi.\n- Defisit kalori bertahap sangat disarankan.\n- Hindari makan berat 3 jam sebelum tidur.")
                else:
                    st.error("- Cut down on high glycemic index carbohydrates.\n- Gradual caloric deficit is highly recommended.\n- Avoid heavy meals 3 hours before sleep.")
            elif "Overweight" in st.session_state['res_asli']:
                if lang == "Bahasa Indonesia":
                    st.warning("- Kurangi porsi makanan manis dan gorengan.\n- Perbanyak porsi protein tanpa lemak.")
                else:
                    st.warning("- Reduce sugary foods and fried items.\n- Increase lean protein portions.")
            else:
                if lang == "Bahasa Indonesia":
                    st.success("- Diet Anda berada pada jalur seimbang.\n- Pastikan asupan makronutrisi tetap terpenuhi.")
                else:
                    st.success("- Your diet is balanced.\n- Ensure your macronutrient intake remains fulfilled.")
        
        with c2:
            st.subheader(ui["sport_title"])
            if "Obesity" in st.session_state['res_asli'] or "Overweight" in st.session_state['res_asli']:
                if lang == "Bahasa Indonesia":
                    st.error("- Mulai dengan olahraga *low-impact* (jalan cepat/berenang).\n- Targetkan minimal **7000 Langkah/hari**.\n- Kurangi duduk terlalu lama.")
                else:
                    st.error("- Start with low-impact exercises (brisk walking/swimming).\n- Aim for at least **7000 steps/day**.\n- Reduce sedentary behavior.")
            else:
                if lang == "Bahasa Indonesia":
                    st.success("- Kombinasi latihan kardio dan beban 3-4 kali seminggu.\n- Target **10.000 Langkah/hari**.")
                else:
                    st.success("- Combine cardio and strength training 3-4 times a week.\n- Target **10.000 steps/day**.")

# ==========================================
# TAB 3: VISUALISASI DATASET LATIHAN
# ==========================================
with tab3:
    st.subheader(ui["chart_title"])
    g1, g2 = st.columns(2)
    with g1:
        pie_title = "Proporsi Kelas Obesitas" if lang == "Bahasa Indonesia" else "Obesity Class Proportions"
        fig1 = px.pie(df_raw, names='NObeyesdad', title=pie_title, hole=0.3, color_discrete_sequence=px.colors.sequential.Agsunset)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        hist_title = "Distribusi Usia Terhadap Status" if lang == "Bahasa Indonesia" else "Age Distribution by Status"
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title=hist_title, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
