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
# 2. INJEKSI CUSTOM CSS (MEMPERCANTIK UI & KONTRAS WAA)
# ==========================================
st.markdown("""
<style>
    /* Latar belakang utama aplikasi (Gradien Soft Blue-Gray) */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d7e1ec 100%);
    }
    
    /* Mempercantik Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }

    /* Kotak Metrik (Hasil, BMI, Akurasi) bergaya Card/Kartu */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #2e86c1; /* Garis aksen biru */
    }

    /* Memaksa kontras teks agar gelap dan mudah dibaca di latar terang */
    html, body, [class*="css"] {
        color: #1a252f;
    }

    /* Warna judul khusus agar lebih elegan */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Warna Tabs (Tab Menu) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. PROSES LOADING INSTAN
# ==========================================
@st.cache_resource
def load_fast_model():
    with open('model_ensemble_signifikan.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

@st.cache_data
def load_data():
    return pd.read_excel("KEL.2 obesitas Projek MCL 2.xlsx", sheet_name=0)

# Efek animasi loading
if 'initialized' not in st.session_state:
    with st.spinner("🔮 Menyeduh ramuan algoritma AI... / Brewing AI algorithms..."):
        time.sleep(1.2)
    st.toast("✨ Selesai! / Done!", icon="🎉")
    st.session_state['initialized'] = True

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
# 4. SIDEBAR & MENU NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=90)
    st.title("Menu / Nav")
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    if lang == "Bahasa Indonesia":
        st.subheader("💧 Target Air Harian")
        bb_calc = st.number_input("Berat Badan Anda (kg)", 30, 200, 60)
        st.info(f"Kebutuhan Hidrasi Minimum: **{bb_calc * 0.033:.2f} Liter/hari**")
    else:
        st.subheader("💧 Daily Water Target")
        bb_calc = st.number_input("Your Weight (kg)", 30, 200, 60)
        st.info(f"Minimum Hydration Need: **{bb_calc * 0.033:.2f} Liters/day**")
        
    st.divider()
    st.caption("🏆 AI Project Kelompok 2 - Jamsix")

# ==========================================
# 5. KAMUS BAHASA (DICTIONARY TERJEMAHAN)
# ==========================================
# Fungsi untuk menerjemahkan class (output model) ke bahasa yang dipilih
def terjemahkan_hasil_ai(hasil_asli, bahasa):
    kamus_indo = {
        'Insufficient_Weight': 'KEKURANGAN BERAT BADAN',
        'Normal_Weight': 'BERAT BADAN NORMAL',
        'Overweight_Level_I': 'KELEBIHAN BERAT BADAN (Tingkat I)',
        'Overweight_Level_II': 'KELEBIHAN BERAT BADAN (Tingkat II)',
        'Obesity_Type_I': 'OBESITAS (Tipe I)',
        'Obesity_Type_II': 'OBESITAS (Tipe II)',
        'Obesity_Type_III': 'OBESITAS ekstrim (Tipe III)'
    }
    kamus_inggris = {
        'Insufficient_Weight': 'INSUFFICIENT WEIGHT',
        'Normal_Weight': 'NORMAL WEIGHT',
        'Overweight_Level_I': 'OVERWEIGHT (Level I)',
        'Overweight_Level_II': 'OVERWEIGHT (Level II)',
        'Obesity_Type_I': 'OBESITY (Type I)',
        'Obesity_Type_II': 'OBESITY (Type II)',
        'Obesity_Type_III': 'EXTREME OBESITY (Type III)'
    }
    
    if bahasa == "Bahasa Indonesia":
        return kamus_indo.get(hasil_asli, hasil_asli.replace('_', ' ').upper())
    else:
        return kamus_inggris.get(hasil_asli, hasil_asli.replace('_', ' ').upper())

# Kamus UI Form
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
        "btn": "🚀 ANALISIS SEKARANG",
        "load1": "🧠 AI sedang menganalisis kebiasaan tubuhmu...",
        "load2": "🎯 Menghitung kecenderungan metabolisme...",
        "load3": "✅ Diagnosis Komputasi Selesai!",
        "res_title": "📊 Hasil Analisis Medis",
        "res_status": "STATUS KESEHATAN",
        "lbl_diag": "Hasil Diagnosis Akhir", "lbl_bmi": "Nilai BMI", "lbl_conf": "Tingkat Keyakinan AI",
        "tab2_warn": "👈 Silakan lakukan analisis pada tab pertama terlebih dahulu.",
        "tab2_title": "📋 Rekomendasi Medis untuk Status:",
        "food_title": "🥗 Panduan Pola Konsumsi",
        "sport_title": "🚴 Panduan Aktivitas Fisik",
        "chart_title": "📊 Statistik Representatif Grafik Dataset"
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
        "btn": "🚀 RUN DIAGNOSTIC NOW",
        "load1": "🧠 AI is analyzing your body metrics...",
        "load2": "🎯 Calculating metabolic tendencies...",
        "load3": "✅ Computational Diagnosis Complete!",
        "res_title": "📊 Medical Analysis Result",
        "res_status": "HEALTH STATUS",
        "lbl_diag": "Final Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Confidence Level",
        "tab2_warn": "👈 Please run the analysis on the first tab first.",
        "tab2_title": "📋 Medical Recommendations for:",
        "food_title": "🥗 Diet & Nutrition Guide",
        "sport_title": "🚴 Physical Activity Guide",
        "chart_title": "📊 Dataset Statistics Representation"
    }

st.title(ui["title"])
st.markdown(f"**{ui['subtitle']}**")
st.divider()

tab1, tab2, tab3 = st.tabs(ui["tabs"])

# ==========================================
# TAB 1: FORM INPUT UTAMA
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
            time.sleep(0.8)
            status.update(label=ui["load2"], state="running")
            time.sleep(0.5)
            status.update(label=ui["load3"], state="complete")
            
        bmi = weight / (height**2)
        
        # Mapping gender kembali ke format model (Female/Male)
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

        # === TRANSLASI HASIL AI KE BAHASA YANG DIPILIH ===
        hasil_terjemahan = terjemahkan_hasil_ai(hasil_prediksi_asli, lang)

        # Simpan state untuk tab 2
        st.session_state['res_asli'] = hasil_prediksi_asli
        st.session_state['res_terjemahan'] = hasil_terjemahan

        st.markdown("---")
        st.subheader(ui["res_title"])

        # Alert Box
        if "Obesity" in hasil_prediksi_asli:
            st.error(f"⚠️ **{ui['res_status']}: {hasil_terjemahan}**")
        elif "Overweight" in hasil_prediksi_asli or "Insufficient" in hasil_prediksi_asli:
            st.warning(f"⚠️ **{ui['res_status']}: {hasil_terjemahan}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {hasil_terjemahan}**")
            st.balloons()

        st.write("")
        
        # Layout Metrik Berjejer 3 (Dengan CSS Custom agar berbentuk kartu)
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label=ui["lbl_diag"], value=hasil_terjemahan)
        with res_col2:
            st.metric(label=ui["lbl_bmi"], value=f"{bmi:.2f}")
        with res_col3:
            st.metric(label=ui["lbl_conf"], value=f"{confidence_score:.2f}%")
        
        st.write("")
        # Progress Bar visual untuk tingkat kepercayaan AI
        st.progress(int(confidence_score) / 100)

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
# TAB 3: VISUALISASI
# ==========================================
with tab3:
    st.subheader(ui["chart_title"])
    g1, g2 = st.columns(2)
    with g1:
        # Chart 1
        pie_title = "Proporsi Kelas Obesitas" if lang == "Bahasa Indonesia" else "Obesity Class Proportions"
        fig1 = px.pie(df_raw, names='NObeyesdad', title=pie_title, hole=0.3, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        # Chart 2
        hist_title = "Distribusi Usia Terhadap Status" if lang == "Bahasa Indonesia" else "Age Distribution by Status"
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title=hist_title, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
