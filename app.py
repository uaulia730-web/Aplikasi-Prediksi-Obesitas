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
# 1. KONFIGURASI & STYLE CSS CUSTOM
# ==========================================
st.set_page_config(page_title="Obesity AI Advisor", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    /* Background Utama */
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    /* Sidebar Emerald Green */
    [data-testid="stSidebar"] { background-color: #064e3b; color: white; border-right: 5px solid #d4af37; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; background-color: white; border-radius: 15px 15px 0px 0px;
        padding: 10px 30px; font-weight: bold; color: #064e3b; border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(90deg, #10b981, #059669) !important; 
        color: white !important; box-shadow: 0 4px 15px rgba(16,185,129,0.4);
    }

    /* Card/Box Styling */
    .card {
        padding: 30px; border-radius: 25px; background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1); margin-bottom: 25px;
        border-top: 10px solid #d4af37;
    }
    
    /* Button Gold Premium */
    .stButton>button {
        width: 100%; border-radius: 50px; height: 4em;
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: white; font-weight: 800; font-size: 20px; border: none;
        transition: 0.4s ease;
    }
    .stButton>button:hover { transform: translateY(-5px); box-shadow: 0 12px 30px rgba(212,175,55,0.5); }
    </style>
    """, unsafe_allowed_html=True)

# ==========================================
# 2. TRAINING ENGINE (MODEL ENSEMBLE TERBAIK)
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
    st.error("Gagal memuat data! Pastikan file Excel 'KEL.2 obesitas Projek MCL 2.xlsx' sudah diletakkan dalam folder yang sama.")
    st.stop()

# ==========================================
# 3. SIDEBAR & BAHASA
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=120)
    st.title("Menu Utama")
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.subheader("💧 Target Air Harian")
    bb_calc = st.number_input("Berat Badan (kg)", 30, 200, 60)
    st.write(f"Kebutuhan Hidrasi: **{bb_calc * 0.033:.2f} Liter/hari**")
    
    st.divider()
    st.info("AI Project Kelompok 2 - Ensemble Learning")

# Kamus Teks Multibahasa
t = {
    "header": "🥗 Obesity AI Advisor" if lang == "English" else "🥗 Penasihat AI Obesitas",
    "sub": "Smart Health Diagnostic based on Ensemble Machine Learning (98.37% Accuracy)" if lang == "English" else "Diagnostik Kesehatan Cerdas berbasis Ensemble Machine Learning (Akurasi 98.37%)",
    "tab1": "🎯 Prediksi AI", "tab2": "💡 Saran Ahli", "tab3": "📊 Statistik Data",
    "btn": "🚀 ANALISIS SEKARANG"
}

# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================
st.markdown(f"<h1 style='text-align: center; color: #064e3b; font-size: 3.5em;'>{t['header']}</h1>", unsafe_allowed_html=True)
st.markdown(f"<p style='text-align: center; color: #555; font-size: 1.2em;'>{t['sub']}</p>", unsafe_allowed_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# --- TAB 1: FORM INPUT VARIABEL SIGNIFIKAN ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>👤 Profil Fisik (Signifikan)</h3>', unsafe_allowed_html=True)
        gender = st.selectbox("Jenis Kelamin / Gender", ["Female", "Male"])
        age = st.number_input("Usia / Age (Tahun)", 1, 100, 21)
        height = st.number_input("Tinggi Badan / Height (m)", 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input("Berat Badan / Weight (kg)", 10, 250, 60, step=0.5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🍏 Kebiasaan & Gaya Hidup (Signifikan)</h3>', unsafe_allowed_html=True)
        fcvc = st.slider("Frekuensi Konsumsi Sayur / Vegetables (1: Jarang, 2: Kadang, 3: Selalu)", 1.0, 3.0, 2.0, step=1.0)
        ch2o = st.slider("Konsumsi Air Minum / Water Intake (Liter per Hari)", 1.0, 3.0, 2.0, step=0.5)
        faf = st.slider("Aktivitas Fisik harian / Physical Activity (0: Pasif, 3: Sangat Aktif)", 0.0, 3.0, 1.0, step=1.0)
        tue = st.slider("Waktu Penggunaan Gadget / Screen Time (0: Rendah, 2: Tinggi)", 0.0, 2.0, 1.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(t["btn"]):
        bmi = weight / (height**2)
        
        # Transformasi input teks Gender ke angka sesuai encoder
        gender_encoded = encoders['Gender'].transform([gender])[0]
        
        # Membuat Dataframe Masukan terstruktur sesuai struktur Fitur Signifikan model akhir
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

        # Proses Klasifikasi menggunakan Ensemble (Voting Classifier)
        pred = model.predict(input_data)[0]
        final_res = target_classes[int(pred)].replace('_', ' ')
        
        st.session_state['res'] = final_res
        st.session_state['bmi'] = bmi

        st.markdown("---")
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Hasil Diagnosis Berbasis Ensemble AI", final_res)
        with res2:
            st.metric("Skor BMI Kalkulasi", f"{bmi:.2f}")
        st.balloons()

# --- TAB 2: SARAN KLINIS ---
with tab2:
    if 'res' not in st.session_state:
        st.info("Silakan lakukan prediksi terlebih dahulu di tab Prediksi.")
    else:
        st.markdown(f"### 💡 Rekomendasi Kesehatan Untuk Risiko: **{st.session_state['res']}**")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown('<div class="card"><h4>🥗 Intervensi Pola Makan</h4>', unsafe_allowed_html=True)
            if "Obesity" in st.session_state['res']:
                st.write("- Batasi asupan kalori pekat dan prioritaskan makanan dengan densitas energi rendah.")
                st.write("- Pertahankan konsumsi serat alami harian (sayuran dan buah-buahan).")
            else:
                st.write("- Pertahankan pemenuhan gizi seimbang makronutrien harian Anda.")
            st.markdown('</div>', unsafe_allow_html=True)
        with c_s2:
            st.markdown('<div class="card"><h4>🚴 Regulasi Aktivitas Fisik</h4>', unsafe_allowed_html=True)
            steps = 7000 if "Obesity" in st.session_state['res'] else 10000
            st.write(f"- Target aktivitas kardio terukur: Minimal **{steps} Langkah per hari**.")
            st.write("- Batasi perilaku sedentari dan kurangi durasi penggunaan gawai (*screen time*) non-produktif.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: VISUALISASI DATASET ---
with tab3:
    st.subheader("📊 Analisis Visual Distribusi Dataset Asli")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Distribusi Kategori Tingkat Gizi pada Dataset", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Korelasi Komposisi Usia Terhadap Kategori Risiko")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    st.markdown("#### Bagan Referensi Klasifikasi BMI Internasional")
    st.image("https://cdn.pixabay.com/photo/2020/05/18/18/14/bmi-5187843_1280.png", width=700)
