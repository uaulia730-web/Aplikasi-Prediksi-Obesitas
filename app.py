import streamlit as st  # Diperbaiki agar seragam menggunakan st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import time

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA (TAMPILAN INDAH)
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Obesitas - Ensemble Learning", 
    page_icon="🥗", 
    layout="wide"
)

# ==========================================
# 2. PROSES LOADING INSTAN (BACA PKL SECARA CEPAT)
# ==========================================
@st.cache_resource
def load_fast_model():
    with open('model_ensemble_signifikan.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

# Efek animasi loading unik/seru saat web pertama kali dibuka
if 'initialized' not in st.session_state:
    with st.spinner("🔮 Menyeduh ramuan algoritma AI... Tunggu sebentar ya!"):
        time.sleep(1.2)
    with st.spinner("🚀 Menghitung kalori, menimbang akurasi... Hampir siap!"):
        time.sleep(1.0)
    st.toast("✨ Selesai! Selamat datang di masa depan kesehatan digital.", icon="🎉")
    st.session_state['initialized'] = True

try:
    meta_data = load_fast_model()
    model = meta_data['model']
    encoders = meta_data['encoders']
    feature_names = meta_data['features']
    classes = meta_data['classes']
    
    # Membaca data ringan untuk statistik grafik di Tab 3
    df_raw = pd.read_excel("KEL.2 obesitas Projek MCL 2.xlsx", sheet_name=0)
except Exception as e:
    st.error("❌ File pendukung model_ensemble_signifikan.pkl atau file Excel data tidak ditemukan!")
    st.stop()

# ==========================================
# 3. SIDEBAR & MENU NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=90)
    st.title("Menu Navigasi")
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.subheader("💧 Target Air Harian")
    bb_calc = st.number_input("Berat Badan Anda (kg)", 30, 200, 60)
    st.info(f"Kebutuhan Hidrasi Minimum: **{bb_calc * 0.033:.2f} Liter/hari**")
    
    st.divider()
    st.caption("🏆 AI Project Kelompok 2 - Jamsix")

# Kamus Teks Dinamis Berdasarkan Bahasa
if lang == "English":
    title_text = "🥗 Obesity AI Advisor"
    sub_text = "⚡ Instant Health Diagnostic Powered by LightGBM & CatBoost Ensemble (98.37% Accuracy)"
    tab1_title, tab2_title, tab3_title = "🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"
    btn_text = "🚀 RUN DIAGNOSTIC NOW"
else:
    title_text = "🥗 Penasihat AI Obesitas"
    sub_text = "⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning (Akurasi 98.37%)"
    tab1_title, tab2_title, tab3_title = "🎯 Prediksi AI", "💡 Saran Pakar", "📊 Statistik Data"
    btn_text = "🚀 ANALISIS SEKARANG"

# Header Utama Aplikasi
st.title(title_text)
st.markdown(f"*{sub_text}*")
st.divider()

tab1, tab2, tab3 = st.tabs([tab1_title, tab2_title, tab3_title])

# --- TAB 1: FORM INPUT UTAMA ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Profil Fisik")
        gender = st.selectbox("Jenis Kelamin / Gender", ["Female", "Male"])
        age = st.number_input("Usia / Age (Tahun)", 1, 100, 21)
        height = st.number_input("Tinggi Badan / Height (Meter)", 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input("Berat Badan / Weight (Kilogram)", 10.0, 250.0, 60.0, step=0.5)

    with col2:
        st.subheader("🍏 Pola Gaya Hidup")
        fcvc = st.slider("Frekuensi Makan Sayur (1: Jarang, 2: Kadang, 3: Selalu)", 1.0, 3.0, 2.0, step=1.0)
        ch2o = st.slider("Konsumsi Air Minum Harian (Liter)", 1.0, 3.0, 2.0, step=0.5)
        faf = st.slider("Aktivitas Fisik / Olahraga Mingguan (0: Pasif, 3: Sangat Aktif)", 0.0, 3.0, 1.0, step=1.0)
        tue = st.slider("Waktu Penggunaan Layar Gadget (0: Sebentar, 2: Lama Semalaman)", 0.0, 2.0, 1.0, step=1.0)

    st.write("")
    if st.button(btn_text, type="primary", use_container_width=True):
        with st.status("🧠 AI sedang menganalisis kebiasaan tubuhmu...", expanded=True) as status:
            time.sleep(0.8)
            status.update(label="🎯 Menghitung kecenderungan metabolisme...", state="running")
            time.sleep(0.5)
            status.update(label="✅ Diagnosis Kedokteran Komputasi Selesai!", state="complete")
            
        bmi = weight / (height**2)
        
        # Konversi kategori gender ke numeric sesuai pkl
        gender_encoded = encoders['Gender'].transform([gender])[0]
        
        # Membuat DataFrame Input Sinkron dengan urutan pkl
        input_data = pd.DataFrame([{
            'Weight': float(weight),
            'Height': float(height),
            'Age': float(age),
            'FCVC': float(fcvc),
            'TUE': float(tue),
            'Gender': gender_encoded,
            'FAF': float(faf),
            'CH2O': float(ch2o)
        }], columns=feature_names)

        # Melakukan Soft Voting Probability untuk akurasi tinggi
        probabilities = model.predict_proba(input_data)[0]
        prediction_idx = np.argmax(probabilities)
        hasil_prediksi = classes[prediction_idx]
        confidence_score = probabilities[prediction_idx] * 100

        st.session_state['res'] = hasil_prediksi
        st.session_state['bmi'] = bmi

        st.markdown("---")
        st.subheader("📊 Hasil Analisis Medis")
        
        color_map = {
            "Insufficient_Weight": "#3B82F6", "Normal_Weight": "#10B981",
            "Overweight_Level_I": "#F59E0B", "Overweight_Level_II": "#F97316",
            "Obesity_Type_I": "#EF4444", "Obesity_Type_II": "#DC2626", "Obesity_Type_III": "#991B1B"
        }
        bg_color = color_map.get(hasil_prediksi, "#1E3A8A")
        
        st.markdown(f"""
            <div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;'>
                <p style='margin: 0; font-size: 1.1rem; font-weight: 300;'>Status Klasifikasi Pengguna:</p>
                <h2 style='margin: 5px 0 0 0; color: white;'>{hasil_prediksi.replace('_', ' ')}</h2>
                <p style='margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9;'>
                    Tingkat Keyakinan Model Klasifikasi: <b>{confidence_score:.2f}%</b>
                </p>
            </div>
        """, unsafe_allowed_html=True)
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Hasil Diagnosis", value=hasil_prediksi.replace('_', ' '))
        with res_col2:
            st.metric(label="Nilai BMI Anda", value=f"{bmi:.2f}")
        st.balloons()

# --- TAB 2: SARAN KESEHATAN MEDIS ---
with tab2:
    if 'res' not in st.session_state:
        st.info("Silakan lakukan pengujian prediksi terlebih dahulu pada tab pertama.")
    else:
        st.subheader(f"📋 Lembar Rekomendasi Medis untuk Status: {st.session_state['res']}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🥗 Panduan Pola Konsumsi")
            if "Obesity" in st.session_state['res']:
                st.write("- Pangkas konsumsi karbohidrat sederhana berindeks glikemik tinggi.")
                st.write("- Optimalkan konsumsi serat sayuran harian Anda.")
            else:
                st.write("- Diet Anda berada pada jalur seimbang, pertahankan porsi gizi saat ini.")
        with c2:
            st.subheader("🚴 Panduan Aktivitas Fisik")
            steps = 7000 if "Obesity" in st.session_state['res'] else 10000
            st.write(f"- Tingkatkan aktivitas pembakaran kalori harian minimal: **{steps} Langkah**.")
            st.write("- Kendalikan durasi diam (*sedentary behavior*) di depan komputer atau layar telepon.")

# --- TAB 3: VISUALISASI ---
with tab3:
    st.subheader("📊 Statistik Representatif Grafik Dataset")
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Kasus Obesitas", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Demografi Usia Terhadap Status Gizi")
        st.plotly_chart(fig2, use_container_width=True)
