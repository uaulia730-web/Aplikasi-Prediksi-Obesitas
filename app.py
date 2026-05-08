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
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px; background-color: #ffffff;
        border-radius: 12px 12px 0px 0px; padding: 10px 25px;
        font-weight: 600; color: #1e3d33; border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important; 
        color: white !important; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .card {
        padding: 25px; border-radius: 20px; background-color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 25px;
        border: 1px solid #efefef;
    }
    .stButton>button {
        width: 100%; border-radius: 50px; height: 3.5em;
        background: linear-gradient(90deg, #d4af37, #b8860b);
        color: white; font-weight: bold; font-size: 18px; border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 8px 20px rgba(212,175,55,0.4); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. SISTEM BAHASA (DICTIONARY)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2737/2737140.png", width=80)
    lang = st.radio("🌐 Select Language / Pilih Bahasa", ["Bahasa Indonesia", "English"])
    st.divider()

# Kamus Terjemahan
text = {
    "title": "🥗 Obesity AI Advisor" if lang == "English" else "🥗 Penasihat AI Obesitas",
    "subtitle": "Smart Health Diagnostic based on Machine Learning" if lang == "English" else "Diagnostik Kesehatan Cerdas berbasis Machine Learning",
    "tab1": "🎯 Prediction / Prediksi",
    "tab2": "💡 Advice / Saran",
    "tab3": "📚 Knowledge / Pengetahuan",
    "input_header1": "👤 Personal Data / Data Diri",
    "input_header2": "🍽️ Lifestyle / Gaya Hidup",
    "gender": "Gender / Jenis Kelamin",
    "age": "Age / Usia",
    "height": "Height / Tinggi (m)",
    "weight": "Weight / Berat (kg)",
    "family": "Family History of Overweight? / Riwayat Obesitas Keluarga?",
    "favc": "Frequent High Calorie Food? / Sering Makan Kalori Tinggi?",
    "fcvc": "Vegetable Consumption (1-3) / Konsumsi Sayur (1-3)",
    "caec": "Eating Between Meals? / Ngemil di Antara Waktu Makan?",
    "faf": "Physical Activity Frequency (0-3) / Frekuensi Olahraga (0-3)",
    "mtrans": "Main Transportation / Transportasi Utama",
    "btn": "🚀 ANALYZE NOW / ANALISIS SEKARANG",
    "result_header": "AI Diagnosis Result / Hasil Diagnosis AI",
    "bmi_header": "Your BMI Score / Skor BMI Anda",
    "info_tab2": "Please perform a prediction first." if lang == "English" else "Silakan lakukan prediksi terlebih dahulu.",
    "bmi_table": "BMI Table Reference" if lang == "English" else "Referensi Tabel BMI"
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
    st.error("Error: Excel file not found!")
    st.stop()

# ==========================================
# 4. TAMPILAN MULTI-TAB
# ==========================================
st.markdown(f"<h1 style='text-align: center; color: #1e3d33;'>{text['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #666;'>{text['subtitle']}</p>", unsafe_allow_html=True)

t1, t2, t3 = st.tabs([text['tab1'], text['tab2'], text['tab3']])

with t1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="card"><h3>{text["input_header1"]}</h3>', unsafe_allow_html=True)
        gender = st.selectbox(text["gender"], ["Female", "Male"])
        age = st.number_input(text["age"], 1, 100, 21)
        height = st.number_input(text["height"], 1.0, 2.5, 1.65)
        weight = st.number_input(text["weight"], 10, 250, 60)
        family = st.selectbox(text["family"], ["yes", "no"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="card"><h3>{text["input_header2"]}</h3>', unsafe_allow_html=True)
        favc = st.selectbox(text["favc"], ["yes", "no"])
        fcvc = st.slider(text["fcvc"], 1.0, 3.0, 2.0)
        caec = st.selectbox(text["caec"], ["no", "Sometimes", "Frequently", "Always"])
        faf = st.slider(text["faf"], 0.0, 3.0, 1.0)
        mtrans = st.selectbox(text["mtrans"], ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(text["btn"]):
        bmi = weight / (height**2)
        input_data = pd.DataFrame([[gender, age, height, weight, family, favc, fcvc, 2.0, caec, 'no', 2.0, 'no', faf, 1.0, 'no', mtrans]], 
                                columns=feature_names)
        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col].astype(str))

        pred = model.predict(input_data)[0][0]
        final_res = target_classes[int(pred)].replace('_', ' ')
        
        st.session_state['last_result'] = final_res
        
        st.divider()
        res_c1, res_c2 = st.columns(2)
        res_c1.metric(text["result_header"], final_res)
        res_c2.metric(text["bmi_header"], f"{bmi:.2f}")
        st.balloons()

with t2:
    st.subheader(text["tab2"])
    if 'last_result' not in st.session_state:
        st.warning(text["info_tab2"])
    else:
        res = st.session_state['last_result']
        st.info(f"Analysis for: **{res}**")
        # Logika saran dwibahasa bisa ditambahkan di sini sesuai kategori

with t3:
    st.subheader(text["bmi_table"])
    bmi_data = {
        "Category / Kategori": ["Underweight", "Normal", "Overweight", "Obesity I", "Obesity II", "Obesity III"],
        "BMI Range / Rentang": ["< 18.5", "18.5 - 24.9", "25.0 - 29.9", "30.0 - 34.9", "35.0 - 39.9", "> 40.0"]
    }
    st.table(pd.DataFrame(bmi_data))
    
    st.markdown("---")
    st.markdown("### 🧬 Factors / Faktor")
    if lang == "English":
        st.write("Obesity is caused by complex factors including genetics, high calorie intake, and sedentary lifestyle.")
    else:
        st.write("Obesitas disebabkan oleh faktor kompleks termasuk genetika, asupan kalori tinggi, dan kurangnya aktivitas fisik.")



[Image of BMI classification chart]
