import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI & STYLE PREMIUM
# ==========================================
st.set_page_config(page_title="Obesity AI Advisor", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    [data-testid="stSidebar"] { background-color: #1e3d33; color: white; }
    .card {
        padding: 25px; border-radius: 20px; background-color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 25px;
        border: 1px solid #efefef; border-left: 8px solid #2e7d32;
    }
    .stButton>button {
        width: 100%; border-radius: 50px; height: 3.8em;
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: white; font-weight: bold; font-size: 18px; border: none;
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important; 
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. PROSES TRAINING (MENGGANTIKAN TRAIN.PY)
# ==========================================
@st.cache_resource
def build_model():
    # Load data dari Excel yang ada di GitHub kamu
    path = "KEL.2 obesitas Projek MCL 2.xlsx"
    df = pd.read_excel(path, sheet_name=0)

    # Preprocessing (Persis seperti train.py kamu)
    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    # SMOTE & Split (Persis seperti train.py kamu)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # Memakai settingan CatBoost TERBAIK kamu (iterations=1000, lr=0.05, depth=8)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0)
    model.fit(X_train, y_train)
    
    return model, le_dict, X.columns.tolist(), le_dict['NObeyesdad'].classes_.tolist()

# Jalankan mesin AI
try:
    model, encoders, feature_names, target_classes = build_model()
except:
    st.error("Gagal memuat data! Pastikan file Excel 'KEL.2 obesitas Projek MCL 2.xlsx' sudah diupload ke GitHub.")
    st.stop()

# ==========================================
# 3. SISTEM BAHASA & TAMPILAN
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=100)
    lang = st.radio("🌐 Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    st.info("Kelompok 2 - Machine Learning Project")

t = {
    "title": "🥗 Obesity AI Advisor" if lang == "English" else "🥗 Penasihat AI Obesitas",
    "tab1": "🎯 Prediction / Prediksi",
    "tab2": "💡 Advice / Saran",
    "tab3": "📚 Knowledge / Pengetahuan",
    "btn": "🚀 ANALYZE NOW" if lang == "English" else "🚀 ANALISIS SEKARANG"
}

st.markdown(f"<h1 style='text-align: center; color: #1e3d33;'>{t['title']}</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# --- TAB 1: PREDIKSI ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>👤 Data Fisik</h3>', unsafe_allow_html=True)
        gender = st.selectbox("Gender / Jenis Kelamin", ["Female", "Male"])
        age = st.number_input("Age / Usia", 1, 100, 20)
        height = st.number_input("Height / Tinggi (m)", 1.0, 2.5, 1.65)
        weight = st.number_input("Weight / Berat (kg)", 10, 250, 60)
        family = st.selectbox("Family History? / Riwayat Keluarga?", ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🍎 Gaya Hidup</h3>', unsafe_allow_html=True)
        favc = st.selectbox("Sering Makan Kalori Tinggi?", ["yes", "no"])
        fcvc = st.slider("Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
        caec = st.selectbox("Sering Ngemil?", ["no", "Sometimes", "Frequently", "Always"])
        faf = st.slider("Frekuensi Olahraga (0-3)", 0.0, 3.0, 1.0)
        mtrans = st.selectbox("Transportasi Utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(t["btn"]):
        bmi = weight / (height**2)
        # Menyiapkan data untuk prediksi (NCP, SMOKE, CH2O, SCC, TUE, CALC diisi nilai default)
        input_data = pd.DataFrame([[gender, age, height, weight, family, favc, fcvc, 3.0, caec, 'no', 2.0, 'no', faf, 1.0, 'no', mtrans]], 
                                columns=feature_names)
        
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

        pred = model.predict(input_data)[0][0]
        final_res = target_classes[int(pred)].replace('_', ' ')
        st.session_state['res'] = final_res
        st.session_state['bmi'] = bmi

        st.markdown("---")
        res_c1, res_c2 = st.columns(2)
        res_c1.metric("Hasil Diagnosis AI", final_res)
        res_c2.metric("Skor BMI", f"{bmi:.2f}")
        st.balloons()

# --- TAB 2: SARAN ---
with tab2:
    if 'res' not in st.session_state:
        st.warning("Silakan lakukan prediksi terlebih dahulu.")
    else:
        res = st.session_state['res']
        st.success(f"Analisis untuk: **{res}**")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="card"><h4>🥗 Pola Makan</h4>', unsafe_allow_html=True)
            if "Obesity" in res:
                st.write("- Kurangi gula & kalori.\n- Perbanyak serat.")
            else:
                st.write("- Pertahankan gizi seimbang.")
            st.markdown('</div>', unsafe_allow_html=True)
        with s2:
            st.markdown('<div class="card"><h4>🚴 Aktivitas</h4>', unsafe_allow_html=True)
            st.write(f"Target: **{5000 if 'Obesity' in res else 10000} Langkah/hari**")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: PENGETAHUAN ---
with tab3:
    st.subheader("Tabel Referensi BMI")
    st.table(pd.DataFrame({
        "Kategori": ["Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III"],
        "Skor": ["< 18.5", "18.5-24.9", "25.0-29.9", "30.0-34.9", "35.0-39.9", "> 40.0"]
    }))
