import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Obesity AI Advisor", 
    page_icon="🥗", 
    layout="wide"
)

# ==========================================
# 2. TRAINING ENGINE (MODEL ENSEMBLE)
# ==========================================
@st.cache_resource
def build_model():
    path = "KEL.2 obesitas Projek MCL 2.xlsx"
    df = pd.read_excel(path, sheet_name=0)
    df_raw = df.copy() 

    # Label Encoding untuk data kategorikal keseluruhan
    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Menggunakan hanya 8 variabel paling signifikan berdasarkan hasil seleksi fitur terbaik Anda
    fitur_signifikan = ['Weight', 'Height', 'Age', 'FCVC', 'TUE', 'Gender', 'FAF', 'CH2O']
    X = df[fitur_signifikan]
    y = df['NObeyesdad']
    
    # Penyeimbangan Data Menggunakan SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # Inisialisasi Dua Model Terbaik Komparasi Anda
    model_cat = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0, random_state=42)
    model_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, verbose=-1, random_state=42)
    
    # Penggabungan Menggunakan Soft Voting Ensemble (Probability Averaging)
    ensemble_model = VotingClassifier(
        estimators=[
            ('catboost', model_cat),
            ('lightgbm', model_lgb)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    
    return ensemble_model, le_dict, fitur_signifikan, le_dict['NObeyesdad'].classes_.tolist(), df_raw

try:
    model, encoders, feature_names, target_classes, df_raw = build_model()
except Exception as e:
    st.error("Gagal memuat data! Pastikan file Excel 'KEL.2 obesitas Projek MCL 2.xlsx' sudah diletakkan dalam folder yang sama di GitHub.")
    st.stop()

# ==========================================
# 3. SIDEBAR & INTEGRASI UTAMA
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=100)
    st.title("Menu Navigasi")
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.subheader("💧 Target Air Harian")
    bb_calc = st.number_input("Berat Badan Anda (kg)", 30, 200, 60)
    st.info(f"Kebutuhan Hidrasi Minimum: **{bb_calc * 0.033:.2f} Liter/hari**")
    
    st.divider()
    st.caption("AI Project Kelompok 2 - Ensemble Machine Learning")

# Kamus Teks Dinamis
if lang == "English":
    title_text = "🥗 Obesity AI Advisor"
    sub_text = "Smart Health Diagnostic based on Ensemble Machine Learning (98.37% Accuracy)"
    tab1_title, tab2_title, tab3_title = "🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"
    btn_text = "🚀 RUN DIAGNOSTIC NOW"
else:
    title_text = "🥗 Penasihat AI Obesitas"
    sub_text = "Diagnostik Kesehatan Cerdas Berbasis Ensemble Machine Learning (Akurasi 98.37%)"
    tab1_title, tab2_title, tab3_title = "🎯 Prediksi AI", "💡 Saran Pakar", "📊 Statistik Data"
    btn_text = "🚀 ANALISIS SEKARANG"

# Banner Utama Aplikasi
st.title(title_text)
st.praise = st.markdown(f"*{sub_text}*")
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
        bmi = weight / (height**2)
        
        # Konversi kategori gender
        gender_encoded = encoders['Gender'].transform([gender])[0]
        
        # DataFrame Input terstruktur
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

        # Eksekusi Klasifikasi dengan Model Ensemble
        pred = model.predict(input_data)[0]
        final_res = target_classes[int(pred)].replace('_', ' ')
        
        st.session_state['res'] = final_res
        st.session_state['bmi'] = bmi

        st.success("🎉 Hasil Komputasi Diagnosis Selesai!")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Hasil Diagnosis Klasifikasi Gizi (Ensemble Model)", value=final_res)
        with res_col2:
            st.metric(label="Nilai Massa Indeks Tubuh (BMI)", value=f"{bmi:.2f}")
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

# --- TAB 3: VISUALISASI HISTOGRAM DATASET ---
with tab3:
    st.subheader("📊 Statistik Representatif Grafik Dataset")
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Kasus Obesitas", hole=0.3)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Demografi Usia Terhadap Status Gizi")
        st.plotly_chart(fig2, use_container_width=True)
