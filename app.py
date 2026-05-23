import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA (WAJIB PALING ATAS)
# ==========================================
st.set_page_config(
    page_title="Prediksi Risiko Obesitas", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CSS BRUTE-FORCE (MEMAKSA TAMPILAN TAB)
# ==========================================
st.markdown("""
<style>
    /* Mengimpor Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }

    /* PAKSA BACKGROUND WARNA ABU-ABU KEBIRUAN */
    .stApp { background-color: #f0f4f8 !important; }

    /* ==============================================
       MEMAKSA TAB MENU BERWARNA DAN PUNYA BATAS (TIDAK POLOS)
       ============================================== */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important; 
        border-radius: 15px !important;
        padding: 8px !important;
        gap: 12px !important;
        box-shadow: inset 0px 3px 6px rgba(0,0,0,0.05) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        border: 2px solid #cbd5e1 !important; 
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05) !important;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #475569 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%) !important;
        border: 2px solid #1A2980 !important;
        box-shadow: 0px 6px 15px rgba(38, 208, 206, 0.5) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* ==============================================
       MEMAKSA KOTAK INPUT LEBIH JELAS
       ============================================== */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: #ffffff !important; 
        padding: 20px !important;
        border-radius: 15px !important;
        border: 2px solid #cbd5e1 !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05) !important;
    }
    .stSlider [data-testid="stMarkdownContainer"] p, 
    .stNumberInput [data-testid="stMarkdownContainer"] p, 
    .stSelectbox [data-testid="stMarkdownContainer"] p {
        color: #1A2980 !important;
        font-weight: 800 !important;
    }

    /* TOMBOL UTAMA */
    button[kind="primary"] {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%) !important;
        color: white !important;
        border-radius: 30px !important; 
        border: none !important;
        padding: 15px 30px !important;
        font-weight: 900 !important;
        letter-spacing: 1px !important;
        box-shadow: 0px 8px 20px rgba(255, 75, 43, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI LOADING DATA & CACHING
# ==========================================
@st.cache_resource
def load_fast_model():
    with open('model_ensemble_signifikan.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data

@st.cache_data
def load_data():
    return pd.read_excel("KEL.2 obesitas Projek MCL 2.xlsx", sheet_name=0)

try:
    meta_data = load_fast_model()
    model = meta_data['model']           
    encoders = meta_data['encoders']     
    feature_names = meta_data['features']
    classes = meta_data['classes']       
    df_raw = load_data()
except Exception as e:
    st.error("❌ File 'model_ensemble_signifikan.pkl' atau Excel tidak ditemukan!")
    st.stop() 

# ==========================================
# 4. SIDEBAR MENU
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; font-weight: 900;'>⚙️ Navigasi</h2>", unsafe_allow_html=True)
    lang = st.radio("Pilih Bahasa / Language", ["Bahasa Indonesia", "English"])
    st.divider()
    
    st.markdown("### 💧 Target Air Harian" if lang == "Bahasa Indonesia" else "### 💧 Daily Water Target")
    lbl_bb = "Berat Badan Anda (kg)" if lang == "Bahasa Indonesia" else "Your Weight (kg)"
    bb_calc = st.number_input(lbl_bb, 30, 200, 60, key="water_calc")
    
    hidrasi_air = bb_calc * 0.033
    lbl_hidrasi = "Kebutuhan Hidrasi Minimum" if lang == "Bahasa Indonesia" else "Minimum Hydration Need"
    lbl_liter = "Liter/hari" if lang == "Bahasa Indonesia" else "Liters/day"
    
    st.markdown(f"""
    <div style="background: #e2e8f0; padding: 15px; border-radius: 12px; border: 2px solid #94a3b8; text-align: center;">
        <p style="margin:0; font-size: 14px; font-weight: 700; color:#475569;">{lbl_hidrasi}</p>
        <h2 style="margin: 5px 0 0 0; font-weight: 900; color:#1A2980;">{hidrasi_air:.2f} <span style="font-size:14px;">{lbl_liter}</span></h2>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. HEADER BANNER APLIKASI 
# ==========================================
st.markdown("""
<div style="background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
            padding: 40px 20px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0px 15px 30px rgba(255, 75, 43, 0.4);
            margin: 10px 0px 30px 0px;
            border: 4px solid #ffffff;">
    <h1 style="color: #ffffff !important; font-weight: 900; font-size: 3.5rem !important; margin: 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); font-family: sans-serif;">
        🩺 Penasihat AI Obesitas
    </h1>
    <p style="color: #ffffff !important; font-size: 1.2rem; font-weight: 600; margin-top: 15px; background: rgba(0,0,0,0.2); display: inline-block; padding: 8px 20px; border-radius: 30px;">
        ⚡ Diagnostik Kesehatan Instan Berbasis Ensemble Machine Learning ⚡
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. TRANSLASI KELAS UI
# ==========================================
def terjemahkan_hasil_ai(hasil_asli, bahasa):
    kamus_indo = {
        'Insufficient_Weight': 'KEKURANGAN BERAT BADAN',
        'Normal_Weight': 'BERAT BADAN NORMAL',
        'Overweight_Level_I': 'KELEBIHAN BERAT BADAN (Tingkat I)',
        'Overweight_Level_II': 'KELEBIHAN BERAT BADAN (Tingkat II)',
        'Obesity_Type_I': 'OBESITAS (Tipe I)',
        'Obesity_Type_II': 'OBESITAS (Tipe II)',
        'Obesity_Type_III': 'OBESITAS EKSTRIM (Tipe III)'
    }
    if bahasa == "Bahasa Indonesia":
        return kamus_indo.get(hasil_asli, hasil_asli.replace('_', ' ').upper())
    return hasil_asli.replace('_', ' ').upper()

if lang == "Bahasa Indonesia":
    ui = {
        "tabs": ["🎯 Prediksi AI", "💡 Saran Pakar", "📊 Statistik Data"],
        "phys": "👤 Profil Fisik", "gender_lbl": "Jenis Kelamin", "gender_opt": ["Perempuan", "Laki-laki"],
        "age": "Usia (Tahun)", "height": "Tinggi Badan (Meter)", "weight": "Berat Badan (Kilogram)",
        "life": "🍏 Pola Gaya Hidup",
        "fcvc": "Frekuensi Makan Sayur (1: Jarang, 3: Selalu)", "ch2o": "Air Minum (Liter)",
        "faf": "Aktivitas Fisik (0: Pasif, 3: Aktif)", "tue": "Layar Gadget (0: Sebentar, 2: Lama)",
        "btn": "🚀 JALANKAN DIAGNOSIS AI",
        "res_title": "📊 Hasil Analisis Medis", "res_status": "STATUS",
        "lbl_diag": "Hasil Diagnosis", "lbl_bmi": "Nilai BMI", "lbl_conf": "Akurasi AI",
        "tab2_title": "📋 Rekomendasi Medis untuk Status:",
        "goto_expert_title": "💡 Langkah Selanjutnya",
        "goto_expert_desc": "Klik tab <b>Saran Pakar</b> di bagian atas untuk melihat panduan gizi lengkap, aktivitas fisik, dan referensi medis khusus untuk Anda!"
    }
else:
    ui = {
        "tabs": ["🎯 AI Prediction", "💡 Expert Advice", "📊 Data Statistics"],
        "phys": "👤 Physical Profile", "gender_lbl": "Gender", "gender_opt": ["Female", "Male"],
        "age": "Age (Years)", "height": "Height (Meters)", "weight": "Weight (Kilograms)",
        "life": "🍏 Lifestyle Habits",
        "fcvc": "Veggies (1: Rare, 3: Always)", "ch2o": "Water (Liters)",
        "faf": "Exercise (0: Passive, 3: Active)", "tue": "Screen Time (0: Short, 2: Long)",
        "btn": "🚀 RUN AI DIAGNOSTIC",
        "res_title": "📊 Medical Analysis Result", "res_status": "STATUS",
        "lbl_diag": "Diagnosis", "lbl_bmi": "BMI Value", "lbl_conf": "AI Accuracy",
        "tab2_title": "📋 Medical Recommendations for:",
        "goto_expert_title": "💡 Next Steps",
        "goto_expert_desc": "Click the <b>Expert Advice</b> tab above to view complete nutrition guides, physical activities, and tailored medical references!"
    }

tab1, tab2, tab3 = st.tabs(ui["tabs"])

# ==========================================
# TAB 1: FORM INPUT UTAMA & PREDIKSI
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['phys']}</h3>", unsafe_allow_html=True)
        gender_input = st.selectbox(ui["gender_lbl"], ui["gender_opt"])
        age = st.number_input(ui["age"], 1, 100, 21)
        height = st.number_input(ui["height"], 1.0, 2.5, 1.65, step=0.01)
        weight = st.number_input(ui["weight"], 10.0, 250.0, 60.0, step=0.5)

    with col2:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['life']}</h3>", unsafe_allow_html=True)
        fcvc = st.slider(ui["fcvc"], 1.0, 3.0, 2.0, step=1.0)
        ch2o = st.slider(ui["ch2o"], 1.0, 3.0, 2.0, step=0.5)
        faf = st.slider(ui["faf"], 0.0, 3.0, 1.0, step=1.0)
        tue = st.slider(ui["tue"], 0.0, 2.0, 1.0, step=1.0)

    st.write("")
    
    if st.button(ui["btn"], type="primary", use_container_width=True):
        bmi = weight / (height**2)
        
        gender_to_model = "Female" if gender_input in ["Perempuan", "Female"] else "Male"
        gender_encoded = encoders['Gender'].transform([gender_to_model])[0]
        
        input_data = pd.DataFrame([{
            'Weight': float(weight), 'Height': float(height), 'Age': float(age),
            'FCVC': float(fcvc), 'TUE': float(tue), 'Gender': gender_encoded,
            'FAF': float(faf), 'CH2O': float(ch2o)
        }], columns=feature_names)

        probabilities = model.predict_proba(input_data)[0]
        prediction_idx = np.argmax(probabilities)
        hasil_prediksi_asli = classes[prediction_idx]
        confidence_score = probabilities[prediction_idx] * 100

        hasil_terjemahan = terjemahkan_hasil_ai(hasil_prediksi_asli, lang)

        st.session_state['res_asli'] = hasil_prediksi_asli
        st.session_state['res_terjemahan'] = hasil_terjemahan
        st.session_state['bmi'] = bmi
        st.session_state['conf'] = confidence_score

    # === TAMPILKAN HASIL ===
    if 'res_terjemahan' in st.session_state:
        st.markdown("---")
        st.markdown(f"<div style='background:#ffffff; padding:15px; border-radius:15px; border: 3px solid #1A2980; text-align:center; box-shadow: 0px 5px 15px rgba(0,0,0,0.1);'><h2 style='color: #1A2980; font-weight:900; margin:0;'>{ui['res_title']}</h2></div>", unsafe_allow_html=True)
        st.write("")

        if "Obesity" in st.session_state['res_asli']:
            st.error(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        elif "Overweight" in st.session_state['res_asli'] or "Insufficient" in st.session_state['res_asli']:
            st.warning(f"⚠️ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")
        else:
            st.success(f"✅ **{ui['res_status']}: {st.session_state['res_terjemahan']}**")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric(label=ui["lbl_diag"], value=st.session_state['res_terjemahan'])
        res_col2.metric(label=ui["lbl_bmi"], value=f"{st.session_state['bmi']:.2f}")
        res_col3.metric(label=ui["lbl_conf"], value=f"{st.session_state['conf']:.2f}%")
        
        st.write("")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = st.session_state['bmi'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 45], 'tickwidth': 2, 'tickcolor': "black"},
                'bar': {'color': "rgba(0,0,0,0.4)", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 18.5], 'color': "#3498db", 'name': 'Kurus'}, 
                    {'range': [18.5, 24.9], 'color': "#2ecc71"}, 
                    {'range': [24.9, 29.9], 'color': "#f1c40f"}, 
                    {'range': [29.9, 45], 'color': "#e74c3c"}],  
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 0.85,
                    'value': st.session_state['bmi']}
            }))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # --- FITUR BARU: BANNER PEMBERITAHUAN MENCOLOK ---
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0px 5px 15px rgba(38, 208, 206, 0.4);">
            <h3 style="margin: 0; color: white !important; font-weight: 800;">{ui['goto_expert_title']}</h3>
            <p style="margin: 8px 0 0 0; font-size: 16px; color: white !important;">{ui['goto_expert_desc']}</p>
        </div>
        """, unsafe_allow_html=True)

        laporan_teks = f"""=====================================
LAPORAN DIAGNOSTIK KESEHATAN AI
=====================================
Status Akhir  : {st.session_state['res_terjemahan']}
Nilai BMI     : {st.session_state['bmi']:.2f}
Akurasi Mesin : {st.session_state['conf']:.2f}%

*Catatan: Dokumen ini dicetak otomatis oleh sistem Penasihat AI Obesitas.
Untuk rekomendasi diet, ikuti arahan dari tenaga medis profesional.
====================================="""
        
        st.download_button(
            label="📥 Unduh Laporan Medis (TXT)",
            data=laporan_teks,
            file_name="Laporan_Kesehatan_AI.txt",
            mime="text/plain",
            type="primary"
        )

# ==========================================
# TAB 2: SARAN KESEHATAN MEDIS
# ==========================================
with tab2:
    if 'res_terjemahan' not in st.session_state:
        st.info("👈 Silakan lakukan prediksi pada tab pertama terlebih dahulu.")
    else:
        st.markdown(f"<h3 style='color: #1A2980; font-weight:800;'>{ui['tab2_title']} {st.session_state['res_terjemahan']}</h3>", unsafe_allow_html=True)
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🥗 Panduan Pola Konsumsi")
            if "Obesity" in st.session_state['res_asli']:
                st.error("- Defisit kalori bertahap sangat disarankan.\n- Pangkas karbohidrat berindeks glikemik tinggi.")
            elif "Overweight" in st.session_state['res_asli']:
                st.warning("- Kurangi porsi makanan manis dan gorengan.\n- Perbanyak protein tanpa lemak.")
            else:
                st.success("- Diet Anda seimbang.\n- Pastikan asupan makronutrisi tetap terpenuhi.")
        
        with c2:
            st.markdown("#### 🚴 Panduan Aktivitas Fisik")
            if "Obesity" in st.session_state['res_asli'] or "Overweight" in st.session_state['res_asli']:
                st.error("- Mulai olahraga *low-impact* (berenang/jalan kaki).\n- Targetkan **7000 Langkah/hari**.")
            else:
                st.success("- Kombinasi latihan kardio & beban 3 kali seminggu.\n- Target **10.000 Langkah/hari**.")

        st.write("")
        st.markdown("### 🔗 Referensi Medis Khusus Untuk Anda")
        
        if "Insufficient" in st.session_state['res_asli']:
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/cara-menambah-berat-badan" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/healthy-lifestyle/nutrition-and-healthy-eating/expert-answers/underweight/faq-20058429"
            link_text = "📖 Panduan Sehat Menaikkan Berat Badan"
            bg_color = "linear-gradient(135deg, #f6d365, #fda085)" 
        elif "Normal" in st.session_state['res_asli']:
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/pola-hidup-sehat" if lang == "Bahasa Indonesia" else "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
            link_text = "📖 Tips Mempertahankan Gaya Hidup Sehat"
            bg_color = "linear-gradient(135deg, #11998e, #38ef7d)" 
        else: 
            link_url = "https://www.siloamhospitals.com/informasi-siloam/artikel/cara-diet-sehat-dan-cepat" if lang == "Bahasa Indonesia" else "https://www.mayoclinic.org/healthy-lifestyle/weight-loss/in-depth/weight-loss/art-20047752"
            link_text = "📖 Panduan Diet Defisit Kalori & Turun BB"
            bg_color = "linear-gradient(135deg, #FF416C, #FF4B2B)" 
            
        st.markdown(f"""
        <a href="{link_url}" target="_blank" style="text-decoration: none;">
            <div style="background: {bg_color}; padding: 15px; border-radius: 12px; text-align: center; color: white; font-weight: 800; font-size: 16px; box-shadow: 0px 5px 15px rgba(0,0,0,0.15); border: 2px solid white;">
                {link_text}
            </div>
        </a>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 3: VISUALISASI DATASET LATIHAN
# ==========================================
with tab3:
    st.markdown("<h3 style='color: #1A2980; font-weight:800;'>📊 Statistik Representatif Dataset</h3>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1:
        fig1 = px.pie(df_raw, names='NObeyesdad', title="Proporsi Kelas Obesitas", hole=0.3, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig1, use_container_width=True)
    with g2:
        fig2 = px.histogram(df_raw, x="Age", color="NObeyesdad", title="Distribusi Usia", color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
