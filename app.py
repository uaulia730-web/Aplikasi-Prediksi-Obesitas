import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Obesity AI Advisor", page_icon="🥗", layout="wide")

# Fungsi untuk Training Otomatis (Hanya jalan 1x)
@st.cache_resource
def initial_training():
    # URL file excel kamu di github (Pastikan file excel sudah diupload ke repo github yang sama)
    # Jika file excel ada di folder yang sama di github, cukup tulis namanya
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

# Jalankan proses belajar
try:
    model, encoders, feature_names, target_classes = initial_training()
    st.sidebar.success("✅ AI Brain Ready!")
except:
    st.error("❌ File Excel belum ditemukan di GitHub. Tolong upload file 'KEL.2 obesitas Projek MCL 2.xlsx' ke GitHub kamu.")
    st.stop()

# ==========================================
# 2. TAMPILAN WEB (SAMA SEPERTI SEBELUMNYA)
# ==========================================
st.title("🥗 Obesity AI Advisor (Live Edition)")
st.write("Sistem ini belajar langsung dari data Excel Anda untuk memberikan prediksi akurat.")

# Form Input
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 1, 100, 20)
    height = st.number_input("Height (m)", 1.0, 2.5, 1.65)
    weight = st.number_input("Weight (kg)", 10, 250, 60)
    family = st.selectbox("Family History Overweight?", ["yes", "no"])

with col2:
    favc = st.selectbox("High Calorie Food?", ["yes", "no"])
    fcvc = st.slider("Vegetables Consumption", 1.0, 3.0, 2.0)
    caec = st.selectbox("Eating between Meals?", ["no", "Sometimes", "Frequently", "Always"])
    faf = st.slider("Physical Activity", 0.0, 3.0, 1.0)
    mtrans = st.selectbox("Transportation", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

if st.button("🚀 ANALYZE NOW"):
    bmi = weight / (height**2)
    # Buat data dummy untuk kolom yang tidak ada di input sederhana
    input_data = pd.DataFrame([[gender, age, height, weight, family, favc, fcvc, 2.0, caec, 'no', 2.0, 'no', faf, 1.0, 'no', mtrans]], 
                            columns=feature_names)
    
    # Encoding
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col].astype(str))

    pred = model.predict(input_data)[0][0]
    final_res = target_classes[int(pred)]

    st.success(f"### Result: {final_res}")
    st.info(f"Your BMI: {bmi:.2f}")
