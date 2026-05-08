import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI & STYLE CSS
# ==========================================
st.set_page_config(page_title="Obesity AI Advisor", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    [data-testid="stSidebar"] { background-color: #064e3b; color: white; border-right: 5px solid #d4af37; }
    .card {
        padding: 30px; border-radius: 25px; background-color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1); margin-bottom: 25px;
        border-top: 10px solid #d4af37;
    }
    .stButton>button {
        width: 100%; border-radius: 50px; height: 3.5em;
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: white; font-weight: 800; border: none;
    }
    /* Warna teks sidebar agar putih bersih */
    .st-emotion-cache-16q9sum p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. TRAINING ENGINE (CACHED)
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
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0)
    model.fit(X_train, y_train)
    return model, le_dict, X.columns.tolist(), le_dict['NObeyesdad'].classes_.tolist(), df_raw

model, encoders, feature_names, target_classes, df_raw = build_model()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=120)
    st.markdown("<h2 style='text-align: center; color: white;'>Menu Utama</h2>", unsafe_allow_html=True)
    
    # Navigasi Halaman
    menu = st.radio(
        "Pilih Halaman:",
        ["🎯 Prediksi AI", "💡 Saran Kesehatan", "📚 Pengetahuan Umum"],
        index=0
    )
    
    st.divider()
    lang = st.radio("🌐 Bahasa / Language", ["Indonesia", "English"])
    
    st.divider()
    st.subheader("💧 Target Air")
    bb_calc = st.number_input("Berat (kg)", 30, 200, 60)
    st.write(f"Kebutuhan: **{bb_calc * 0.033:.2f} L/hari**")

# ==========================================
# 4. LOGIKA HALAMAN
# ==========================================

# --- HALAMAN 1: PREDIKSI ---
if menu == "🎯 Prediksi AI":
    st.markdown(f"<h1 style='color: #064e3b;'>🎯 Analisis Status Obesitas</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h3>👤 Profil</h3>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Usia", 1, 100, 21)
        height = st.number_input("Tinggi (m)", 1.0, 2.5, 1.65)
        weight = st.number_input("Berat (kg)", 10, 250, 60)
        family = st.selectbox("Riwayat Keluarga?", ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🍏 Gaya Hidup</h3>', unsafe_allow_html=True)
        favc = st.selectbox("Suka Kalori Tinggi?", ["yes", "no"])
        fcvc = st.slider("Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
        caec = st.selectbox("Sering Ngemil?", ["no", "Sometimes", "Frequently", "Always"])
        faf = st.slider("Olahraga (0-3)", 0.0, 3.0, 1.0)
        mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 ANALISIS SEKARANG"):
        bmi = weight / (height**2)
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
        r1, r2 = st.columns(2)
        r1.metric("Hasil Diagnosis AI", final_res)
        r2.metric("Skor BMI", f"{bmi:.2f}")
        st.balloons()

# --- HALAMAN 2: SARAN ---
elif menu == "💡 Saran Kesehatan":
    st.markdown(f"<h1 style='color: #064e3b;'>💡 Saran Ahli</h1>", unsafe_allow_html=True)
    if 'res' not in st.session_state:
        st.info("Silakan lakukan prediksi terlebih dahulu di menu Prediksi AI.")
    else:
        st.success(f"Analisis untuk: **{st.session_state['res']}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="card"><h4>🥗 Nutrisi</h4>', unsafe_allow_html=True)
            if "Obesity" in st.session_state['res']:
                st.write("- Kurangi gula & karbohidrat.\n- Perbanyak serat.")
            else:
                st.write("- Pertahankan gizi seimbang.")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>🚴 Aktivitas</h4>', unsafe_allow_html=True)
            st.write(f"- Target: **{10000 if 'Normal' in st.session_state['res'] else 5000} langkah/hari**.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- HALAMAN 3: PENGETAHUAN UMUM ---
else:
    st.markdown(f"<h1 style='color: #064e3b;'>📚 Pengetahuan & Statistik</h1>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Distribusi Data")
    fig = px.pie(df_raw, names='NObeyesdad', hole=0.4, color_discrete_sequence=px.colors.sequential.Greens_r)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Referensi BMI")
    st.image("https://cdn.pixabay.com/photo/2020/05/18/18/14/bmi-5187843_1280.png", width=800)
    st.markdown('</div>', unsafe_allow_html=True)
