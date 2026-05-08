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
    /* Background & Font */
    .stApp { background-color: #f4f7f6; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #1e3d33; color: white; }
    
    /* Glassmorphism Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; background-color: #ffffff;
        border-radius: 12px 12px 0px 0px; padding: 10px 25px;
        font-weight: 600; color: #1e3d33; border: 1px solid #e0e0e0;
        transition: 0.3s;
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important; 
        color: white !important; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Card Styling */
    .card {
        padding: 25px; border-radius: 20px; background-color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 25px;
        border: 1px solid #efefef; border-left: 8px solid #2e7d32;
    }
    
    /* Premium Button */
    .stButton>button {
        width: 100%; border-radius: 50px; height: 3.8em;
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: white; font-weight: bold; font-size: 18px; border: none;
        box-shadow: 0 4px 15px rgba(212,175,55,0.3);
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(212,175,55,0.4); }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #1b5e20; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SISTEM BAHASA (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=100)
    st.title("Settings / Pengaturan")
    lang = st.radio("🌐 Select Language / Pilih Bahasa", ["Bahasa Indonesia", "English"])
    st.divider()
    st.info("Kelompok 2 - Machine Learning Project")

# Kamus Teks Bilingual
t = {
    "title": "🥗 Obesity AI Advisor" if lang == "English" else "🥗 Penasihat AI Obesitas",
    "sub": "Smart Diagnostic System" if lang == "English" else "Sistem Diagnostik Cerdas",
    "tab1": "🎯 Prediction / Prediksi",
    "tab2": "💡 Advice / Saran",
    "tab3": "📚 Knowledge / Pengetahuan",
    "c1": "👤 Personal Data / Data Fisik",
    "c2": "🍎 Lifestyle / Gaya Hidup",
    "btn": "🚀 ANALYZE NOW / ANALISIS SEKARANG",
    "bmi_ref": "BMI Reference Table" if lang == "English" else "Tabel Referensi BMI"
}

# ==========================================
# 3. TRAINING ENGINE (CACHED)
# ==========================================
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
except:
    st.error("Error: Excel file missing!")
    st.stop()

# ==========================================
# 4. TAMPILAN DASHBOARD
# ==========================================
st.markdown(f"<h1 style='text-align: center; color: #1e3d33; font-size: 3em;'>{t['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #666; font-size: 1.2em;'>{t['sub']}</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# --- TAB 1: PREDIKSI ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="card"><h3>{t["c1"]}</h3>', unsafe_allow_html=True)
        gender = st.selectbox("Gender / Jenis Kelamin", ["Female", "Male"])
        age = st.number_input("Age / Usia", 1, 100, 20)
        height = st.number_input("Height / Tinggi (m)", 1.0, 2.5, 1.65)
        weight = st.number_input("Weight / Berat (kg)", 10, 250, 60)
        family = st.selectbox("Family Overweight History? / Riwayat Obesitas?", ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><h3>{t["c2"]}</h3>', unsafe_allow_html=True)
        favc = st.selectbox("High Calorie Food? / Makanan Kalori Tinggi?", ["yes", "no"])
        fcvc = st.slider("Vegetables / Sayur (1-3)", 1.0, 3.0, 2.0)
        caec = st.selectbox("Snacking? / Kebiasaan Ngemil?", ["no", "Sometimes", "Frequently", "Always"])
        faf = st.slider("Physical Activity / Olahraga (0-3)", 0.0, 3.0, 1.0)
        mtrans = st.selectbox("Main Transportation / Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(t["btn"]):
        bmi = weight / (height**2)
        input_data = pd.DataFrame([[gender, age, height, weight, family, favc, fcvc, 2.0, caec, 'no', 2.0, 'no', faf, 1.0, 'no', mtrans]], 
                                columns=feature_names)
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

        pred = model.predict(input_data)[0][0]
        final_res = target_classes[int(pred)].replace('_', ' ')
        st.session_state['last_result'] = final_res

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Diagnosis", final_res)
        res_col2.metric("BMI Score", f"{bmi:.2f}")
        st.balloons()

# --- TAB 2: SARAN ---
with tab2:
    if 'last_result' not in st.session_state:
        st.warning("⚠️ Please perform a prediction first / Silakan lakukan prediksi terlebih dahulu.")
    else:
        res = st.session_state['last_result']
        st.success(f"### Analysis for / Analisis untuk: {res}")
        
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f'<div class="card"><h4>🥗 Nutrition / Pola Makan</h4>', unsafe_allow_html=True)
            if "Obesity" in res:
                st.write("- Reduce sugar & high-cal food / Kurangi gula & kalori tinggi.")
                st.write("- High protein & fiber diet / Diet tinggi protein & serat.")
            elif "Overweight" in res:
                st.write("- Portion control / Kontrol porsi makan.")
                st.write("- More vegetables / Perbanyak sayuran.")
            else:
                st.write("- Maintain balanced diet / Pertahankan gizi seimbang.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with s2:
            st.markdown(f'<div class="card"><h4>🚴 Activity / Aktivitas Fisik</h4>', unsafe_allow_html=True)
            if "Obesity" in res:
                st.write("- Daily 30m walking / Jalan kaki 30 menit sehari.")
            elif "Overweight" in res:
                st.write("- Cardio 3x per week / Kardio 3x seminggu.")
            else:
                st.write("- Stay active / Tetap aktif bergerak.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: PENGETAHUAN ---
with tab3:
    st.subheader(t["bmi_ref"])
    bmi_df = pd.DataFrame({
        "Category / Kategori": ["Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III"],
        "Range / Rentang": ["< 18.5", "18.5 - 24.9", "25.0 - 29.9", "30.0 - 34.9", "35.0 - 39.9", "> 40.0"]
    })
    st.table(bmi_df)
    
    
    
    st.markdown("""
    ### 📝 Quick Facts / Fakta Cepat:
    - **Genetics:** Family history plays a role / Riwayat keluarga berpengaruh.
    - **Lifestyle:** Sedentary habits increase risk / Gaya hidup kurang gerak meningkatkan risiko.
    - **Sleep:** Poor sleep disrupts hunger hormones / Kurang tidur mengganggu hormon lapar.
    """)
