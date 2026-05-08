#================
#app py
#================
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="Obesity AI Expert", page_icon="🥗", layout="wide")

# CSS Custom untuk UI Premium
st.markdown("""
    <style>
    .stApp { background: #f0f2f5; }
    .card { padding: 20px; border-radius: 15px; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px; border: 1px solid #e0e0e0; }
    .main-title { font-size: 45px; font-weight: 800; color: #1b5e20; text-align: center; margin-bottom: 10px; }
    .stButton>button { width: 100%; border-radius: 30px; height: 3.5em; background: linear-gradient(90deg, #2e7d32, #43a047); color: white; font-weight: bold; border: none; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

# Load Model (Hanya sekali)
@st.cache_resource
def get_model():
    with open('model_final.pkl', 'rb') as f:
        return pickle.load(f)

data_pack = get_model()
model = data_pack['model']
encoders = data_pack['encoders']
features = data_pack['features']
classes = data_pack['classes']

st.markdown("<h1 class='main-title'>🥗 Obesity AI Expert System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Global Health Diagnostic Tool - Powered by CatBoost AI</p>", unsafe_allow_html=True)
st.write("---")

# Input Section
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👤 Bio Profile")
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", 1, 100, 25)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.7)
        weight = st.number_input("Weight (kg)", 10, 250, 70)
        family = st.selectbox("Family History Overweight?", ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🍽️ Eating Habits")
        favc = st.selectbox("High Calorie Food?", ["yes", "no"])
        fcvc = st.slider("Vegetables Consumption", 1.0, 3.0, 2.0)
        ncp = st.slider("Number of Main Meals", 1.0, 4.0, 3.0)
        caec = st.selectbox("Eating between Meals?", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Water Consumption", 1.0, 3.0, 2.0)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏃 Lifestyle")
        smoke = st.selectbox("Smoking?", ["yes", "no"])
        scc = st.selectbox("Calories Monitoring?", ["yes", "no"])
        faf = st.slider("Physical Activity (0-3)", 0.0, 3.0, 1.0)
        tue = st.slider("Tech Usage (0-2)", 0.0, 2.0, 1.0)
        calc = st.selectbox("Alcohol Consumption?", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportation", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 ANALYZE MY HEALTH"):
    bmi = weight / (height**2)
    inputs = {'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight, 'family_history_with_overweight': family, 'FAVC': favc, 'FCVC': fcvc, 'NCP': ncp, 'CAEC': caec, 'SMOKE': smoke, 'CH2O': ch2o, 'SCC': scc, 'FAF': faf, 'TUE': tue, 'CALC': calc, 'MTRANS': mtrans}
    input_df = pd.DataFrame([inputs])[features]

    for col in input_df.columns:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))

    pred = model.predict(input_df)[0][0]
    res_label = classes[int(pred)]

    # Result UI
    st.markdown("---")
    color = "#2ecc71" if "Normal" in res_label else "#f39c12" if "Overweight" in res_label else "#e74c3c"
    
    st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 30px; border-radius: 20px; text-align: center;">
            <h2 style="margin:0;">DIAGNOSIS: {res_label.replace('_', ' ')}</h2>
            <p style="font-size: 20px; margin:0;">BMI Score: {bmi:.2f}</p>
        </div>
    """, unsafe_allow_html=True)

    # Health Tips Database
    st.write("### 💡 Expert Health Advice")
    tips = {
        "Insufficient_Weight": "Tingkatkan asupan kalori sehat dan fokus pada latihan pembentukan otot.",
        "Normal_Weight": "Luar biasa! Pertahankan pola makan seimbang dan tetap aktif secara fisik.",
        "Overweight_Level_I": "Kurangi porsi makan sedikit demi sedikit dan tambahkan aktivitas kardio 30 menit/hari.",
        "Overweight_Level_II": "Batasi asupan gula dan lemak jenuh. Fokus pada konsumsi sayuran (FCVC) lebih banyak.",
        "Obesity_Type_I": "Mulailah program penurunan berat badan yang konsisten. Monitor kalori (SCC) sangat disarankan.",
        "Obesity_Type_II": "Konsultasikan dengan ahli gizi. Batasi makanan cepat saji (FAVC) secara drastis.",
        "Obesity_Type_III": "Prioritas kesehatan tinggi. Kurangi waktu gadget (TUE) dan ganti dengan berjalan kaki atau olahraga air."
    }
    st.info(tips.get(res_label, "Tetap jaga kesehatan dan pola makan teratur."))
    st.balloons()