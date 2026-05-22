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
# 2. PROSES LOADING INSTAN
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
# 3. SIDEBAR & MENU NAVIGASI (DENGAN LOKALISASI BAHASA)
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
# 4. KAMUS BAHASA (DICTIONARY) UNTUK UI
# ==========================================
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
        "res_status": "STATUS",
        "lbl_diag": "Hasil Diagnosis Akhir", "lbl_bmi": "Nilai BMI", "lbl_conf": "Tingkat Keyakinan (Akurasi)",
        "tab2_warn": "👈 Silakan lakukan analisis pada tab pertama terlebih dahulu.",
        "tab2_title": "📋 Rekomendasi Medis untuk:",
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
        "res_status": "STATUS",
        "lbl_diag": "Final Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Confidence Level",
        "tab2_warn": "👈 Please run the analysis on the first tab first.",
        "tab2_title": "📋 Medical Recommendations for:",
        "food_title": "🥗 Diet & Nutrition Guide",
        "sport_title": "🚴 Physical Activity Guide",
        "chart_title": "📊 Dataset Statistics Representation"
    }

st.title(ui["title"])
st.markdown(f"*{ui['subtitle']}*")
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
        hasil_prediksi = classes[prediction_idx]
        confidence_score = probabilities[prediction_idx] * 100

        st.session_state['res'] = hasil_prediksi
        st.session_state['bmi'] = bmi
        st.session_state['conf'] = confidence_score

        # TAMPILAN HASIL SEJAJAR (DI SAMPING)
        st.markdown("---")
        st.subheader(ui["res_title"])

        kategori_format = hasil_prediksi.replace('_', ' ').upper()

        # Layout 3 Kolom Bersebelahan
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label=ui["lbl_diag"], value=kategori_format)
        with res_col2:
            st.metric(label=ui["lbl_bmi"], value=f"{bmi:.2f}")
        with res_col3:
            st.metric(label=ui["lbl_conf"], value=f"{confidence_score:.2f}%")
        
        # Progress Bar visual untuk tingkat kepercayaan AI
        st.progress(int(confidence_score) / 100)

        # Alert Box
        if "Obesity" in hasil_prediksi:
            st.error(f"⚠️ **{ui['res_status']}: {kategori_format}**")
        elif "Overweight" in hasil_prediksi:
            st.warning(f"⚠️ **{ui['res_status']}: {kategori_format}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {kategori_format}**")
            st.balloons()

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS
# ==========================================
with tab2:
    if 'res' not in st.session_state:
        st.info(ui["tab2_warn"])
    else:
        status_sekarang = st.session_state['res'].replace('_', ' ').upper()
        st.subheader(f"{ui['tab2_title']} {status_sekarang}")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(ui["food_title"])
            if "Obesity" in st.session_state['res']:
                if lang == "Bahasa Indonesia":
                    st.error("- Pangkas konsumsi karbohidrat berindeks glikemik tinggi.\n- Defisit kalori bertahap sangat disarankan.\n- Hindari makan berat 3 jam sebelum tidur.")
                else:
                    st.error("- Cut down on high glycemic index carbohydrates.\n- Gradual caloric deficit is highly recommended.\n- Avoid heavy meals 3 hours before sleep.")
            elif "Overweight" in st.session_state['res']:
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
            if "Obesity" in st.session_state['res'] or "Overweight" in st.session_state['res']:
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
        fig1 = px.pie(df_raw, names='NObeyesdad', title=pie_title, hole=0.3, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        # Chart 2
        hist_title = "Distribusi Usia Terhadap Status" if lang == "Bahasa Indonesia" else "Age Distribution by Status"
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title=hist_title)
        st.plotly_chart(fig2, use_container_width=True)
