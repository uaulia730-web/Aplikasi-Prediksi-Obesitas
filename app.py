import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

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
    """, unsafe_allow_html=True)

# ==========================================
# 2. TRAINING ENGINE (SESUAI MODEL KAMU)
# ==========================================
@st.cache_resource
def build_model():
    path = "KEL.2 obesitas Projek MCL 2.xlsx"
    df = pd.read_excel(path, sheet_name=0)
    df_raw = df.copy() 

    le_dict = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']
    
    # SMOTE & Split (Stratify y agar seimbang)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    # Memakai settingan CatBoost TERBAIK kamu
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0)
    model.fit(X_train, y_train)
    
    return model, le_dict, X.columns.tolist(), le_dict['NObeyesdad'].classes_.tolist(), df_raw

try:
    model, encoders, feature_names, target_classes, df_raw = build_model()
except:
    st.error("Gagal memuat data! Pastikan file Excel 'KEL.2 obesitas Projek MCL 2.xlsx' sudah ada di GitHub.")
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
    st.write(f"Kebutuhan: **{bb_calc * 0.033:.2f} Liter/hari**")
    
    st.divider()
    st.info("AI Project Kelompok 2")

# Kamus Teks
t = {
    "header": "🥗 Obesity AI Advisor" if lang == "English" else "🥗 Penasihat AI Obesitas",
    "sub": "Smart Health Diagnostic based on Machine Learning" if lang == "English" else "Diagnostik Kesehatan Cerdas berbasis Machine Learning",
    "tab1": "🎯 Prediksi AI", "tab2": "💡 Saran Ahli", "tab3": "📊 Statistik Data",
    "btn": "🚀 ANALISIS SEKARANG"
}

# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================
st.markdown(f"<h1 style='text-align: center; color: #064e3b; font-size: 3.5em;'>{t['header']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #555; font-size: 1.2em;'>{t['sub']}</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# --- TAB 1: FORM INPUT ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>👤 Profil Fisik</h3>', unsafe_allow_html=True)
        gender = st.selectbox("Jenis Kelamin / Gender", ["Female", "Male"])
        age = st.number_input("Usia / Age", 1, 100, 21)
        height = st.number_input("Tinggi Badan / Height (m)", 1.0, 2.5, 1.65)
        weight = st.number_input("Berat Badan / Weight (kg)", 10, 250, 60)
        family = st.selectbox("Riwayat Keluarga Obesitas? / Family History?", ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🍏 Pola Hidup</h3>', unsafe_allow_html=True)
        favc = st.selectbox("Suka Makanan Berkalori Tinggi? / High Calorie?", ["yes", "no"])
        fcvc = st.slider("Frekuensi Makan Sayur / Vegetables (1-3)", 1.0, 3.0, 2.0)
        caec = st.selectbox("Sering Ngemil? / Snacking?", ["no", "Sometimes", "Frequently", "Always"])
        faf = st.slider("Aktivitas Fisik / Physical Activity (0-3)", 0.0, 3.0, 1.0)
        mtrans = st.selectbox("Transportasi / Transportation", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
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
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Hasil Diagnosis AI", final_res)
        with res2:
            st.metric("Skor BMI Anda", f"{bmi:.2f}")
        st.balloons()

# --- TAB 2: SARAN ---
with tab2:
    if 'res' not in st.session_state:
        st.info("Silakan lakukan prediksi terlebih dahulu di tab Prediksi.")
    else:
        st.markdown(f"### 💡 Saran Untuk: **{st.session_state['res']}**")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.markdown('<div class="card"><h4>🥗 Pola Makan</h4>', unsafe_allow_html=True)
            if "Obesity" in st.session_state['res']:
                st.write("- Kurangi karbohidrat olahan & gula.")
                st.write("- Perbanyak protein & serat.")
            else:
                st.write("- Pertahankan gizi seimbang.")
            st.markdown('</div>', unsafe_allow_html=True)
        with c_s2:
            st.markdown('<div class="card"><h4>🚴 Aktivitas</h4>', unsafe_allow_html=True)
            steps = 5000 if "Obesity" in st.session_state['res'] else 10000
            st.write(f"- Target langkah harian: **{steps} Langkah**.")
            st.write("- Hindari gaya hidup sedentari (kurang gerak).")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: GRAFIK DATA ---
with tab3:
    st.subheader("📊 Analisis Dataset")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Distribusi Kategori Dataset", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col_g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Hubungan Usia & Kategori")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.divider()
    st.markdown("#### Referensi Kategori BMI")
    st.image("https://cdn.pixabay.com/photo/2020/05/18/18/14/bmi-5187843_1280.png", width=700)
