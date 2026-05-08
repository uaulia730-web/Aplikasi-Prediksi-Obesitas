import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI & STYLE
# ==========================================
st.set_page_config(page_title="Obesity AI Advisor", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #ffffff;
        border-radius: 10px 10px 0px 0px; gap: 1px; padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #2e7d32 !important; color: white !important; }
    .card {
        padding: 20px; border-radius: 15px; background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk Training Otomatis
@st.cache_resource
def initial_training():
    path = "KEL.2 obesitas Projek MCL 2.xlsx" 
    df = pd.read_excel(path, sheet_name=0)
    
    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0)
    model.fit(X_train, y_train)
    
    return model, le_dict, X.columns.tolist(), le_dict['NObeyesdad'].classes_.tolist()

try:
    model, encoders, feature_names, target_classes = initial_training()
    st.sidebar.success("✅ AI Engine Ready")
except:
    st.error("❌ File Excel 'KEL.2 obesitas Projek MCL 2.xlsx' tidak ditemukan di GitHub.")
    st.stop()

# ==========================================
# 2. NAVIGASI TAB (3 SEGMEN)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🎯 Prediksi AI", "💡 Saran Kesehatan", "📚 Pengetahuan Umum"])

# ------------------------------------------
# SEGMEN 1: PREDIKSI
# ------------------------------------------
with tab1:
    st.title("🏃 Analisis Status Obesitas")
    st.write("Masukkan data Anda untuk mendapatkan diagnosis instan dari kecerdasan buatan.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            age = st.number_input("Usia", 1, 100, 20)
            height = st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.65)
            weight = st.number_input("Berat Badan (kg)", 10, 250, 60)
            family = st.selectbox("Riwayat Keluarga Obesitas?", ["yes", "no"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            favc = st.selectbox("Suka Makanan Berkalori Tinggi?", ["yes", "no"])
            fcvc = st.slider("Frekuensi Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
            caec = st.selectbox("Kebiasaan Ngemil?", ["no", "Sometimes", "Frequently", "Always"])
            faf = st.slider("Aktivitas Fisik (FAF)", 0.0, 3.0, 1.0)
            mtrans = st.selectbox("Transportasi Utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
            st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 MULAI ANALISIS"):
        bmi = weight / (height**2)
        input_data = pd.DataFrame([[gender, age, height, weight, family, favc, fcvc, 2.0, caec, 'no', 2.0, 'no', faf, 1.0, 'no', mtrans]], 
                                columns=feature_names)
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

        pred = model.predict(input_data)[0][0]
        final_res = target_classes[int(pred)].replace('_', ' ')

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Hasil Diagnosis AI", final_res)
        c2.metric("Skor BMI Anda", f"{bmi:.2f}")
        st.session_state['last_result'] = final_res # Simpan hasil untuk Tab Saran
        st.balloons()

# ------------------------------------------
# SEGMEN 2: SARAN KESEHATAN
# ------------------------------------------
with tab2:
    st.title("💡 Saran Berdasarkan Hasil AI")
    if 'last_result' not in st.session_state:
        st.info("Silakan lakukan prediksi terlebih dahulu di Tab Prediksi.")
    else:
        res = st.session_state['last_result']
        st.success(f"Saran untuk kategori: **{res}**")
        
        col_saran1, col_saran2 = st.columns(2)
        with col_saran1:
            st.subheader("🥗 Pola Makan")
            if "Obesity" in res:
                st.write("- Kurangi asupan kalori harian secara bertahap.\n- Fokus pada protein dan serat tinggi.\n- Hindari minuman manis dan gorengan.")
            elif "Overweight" in res:
                st.write("- Mulai batasi porsi karbohidrat.\n- Perbanyak sayuran pada setiap jam makan.")
            else:
                st.write("- Pertahankan gizi seimbang.\n- Pastikan asupan nutrisi mencukupi kebutuhan harian.")
        
        with col_saran2:
            st.subheader("🚴 Aktivitas Fisik")
            if "Obesity" in res:
                st.write("- Mulai dengan jalan santai 30 menit setiap hari.\n- Konsultasikan dengan pelatih fisik untuk olahraga beban.")
            elif "Overweight" in res:
                st.write("- Tingkatkan frekuensi olahraga menjadi 3-4 kali seminggu.\n- Gabungkan kardio dan latihan beban.")
            else:
                st.write("- Tetap aktif bergerak.\n- Cobalah variasi olahraga baru untuk menjaga massa otot.")

# ------------------------------------------
# SEGMEN 3: PENGETAHUAN UMUM
# ------------------------------------------
with tab3:
    st.title("📚 Pengetahuan Umum Obesitas")
    
    st.markdown("""
    ### Apa itu Obesitas?
    Obesitas adalah kondisi medis berupa penumpukan lemak tubuh berlebih yang dapat berdampak buruk bagi kesehatan.
    
    ### Cara Membaca Skor BMI:
    """)
    
    st.table(pd.DataFrame({
        "Kategori BMI": ["Kurang Berat Badan", "Normal", "Kelebihan Berat Badan", "Obesitas Tipe I", "Obesitas Tipe II", "Obesitas Tipe III"],
        "Rentang Skor": ["< 18.5", "18.5 - 24.9", "25.0 - 29.9", "30.0 - 34.9", "35.0 - 39.9", "> 40.0"]
    }))

    
    
    st.markdown("""
    ### Faktor Risiko Utama:
    1. **Genetika:** Riwayat keluarga sangat berpengaruh.
    2. **Gaya Hidup:** Kurangnya aktivitas fisik (sedenter).
    3. **Pola Makan:** Konsumsi makanan tinggi kalori dan gula secara berlebih.
    4. **Kurang Tidur:** Dapat mengganggu hormon pengatur rasa lapar.
    """)
