import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================================================================
# 1. LOAD DATA & PREPROCESSING AWAL
# =========================================================================
path = r"C:\Users\ASUS\OneDrive\Documents\SEMESTER 6\Mechine Learning\D2 (Uni)\KEL.2 obesitas Projek MCL 2.xlsx"
df = pd.read_excel(path, sheet_name=0)

# Label Encoding untuk kolom teks/kategorikal
le_dict = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X_all = df.drop('NObeyesdad', axis=1)
y_all = df['NObeyesdad']

# SMOTE untuk seluruh fitur awal
smote = SMOTE(random_state=42)
X_res_all, y_res_all = smote.fit_resample(X_all, y_all)

# Split awal untuk mencari feature importance (80:20)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_res_all, y_res_all, test_size=0.2, random_state=42, stratify=y_res_all
)

# =========================================================================
# 2. MENCARI VARIABEL YANG SIGNIFIKAN (FEATURE IMPORTANCE)
# =========================================================================
print("⏳ Menganalisis tingkat signifikansi variabel...")
model_cat_check = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=8, verbose=0, random_state=42)
model_lgb_check = LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8, verbose=-1, random_state=42)

model_cat_check.fit(X_train_f, y_train_f)
model_lgb_check.fit(X_train_f, y_train_f)

# Merata-ratakan pentingnya fitur dari kedua model
importance_cat = model_cat_check.get_feature_importance()
importance_lgb = model_lgb_check.feature_importances_
# Normalisasi skala kepentingan LightGBM agar sebanding dengan CatBoost
importance_lgb_norm = (importance_lgb / importance_lgb.sum()) * 100

final_importance = (importance_cat + importance_lgb_norm) / 2

# Membuat dataframe hasil tingkat kepentingan
feature_imp_df = pd.DataFrame({
    'Variabel': X_all.columns,
    'Tingkat_Signifikan': final_importance
}).sort_values(by='Tingkat_Signifikan', ascending=False).reset_index(drop=True)

print("\n📊 URUTAN VARIABEL BERDASARKAN TINGKAT SIGNIFIKANSI:")
print(feature_imp_df)

# Mengambil variabel yang signifikan (misal: mengambil Top 8 fitur terbaik)
# Jumlah fitur bisa Anda sesuaikan (misal diubah ke 6 atau 10 tergantung kebutuhan)
jumlah_fitur_terpilih = 8
fitur_signifikan = feature_imp_df['Variabel'].head(jumlah_fitur_terpilih).tolist()

print(f"\n✨ Variabel yang dipilih untuk model final: {fitur_signifikan}")

# =========================================================================
# 3. MEMBUAT MODEL BARU BERDASARKAN FITUR SIGNIFIKAN
# =========================================================================
# Menyaring data hanya untuk fitur-fitur yang signifikan saja
X_selected = df[fitur_signifikan]
y_selected = df['NObeyesdad']

# Menerapkan SMOTE kembali pada fitur terpilih
X_res_sel, y_res_sel = smote.fit_resample(X_selected, y_selected)

# Membagi data testing dan training (80:20) menggunakan fitur baru
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_res_sel, y_res_sel, test_size=0.2, random_state=42, stratify=y_res_sel
)

# Inisialisasi Ulang Model Ensemble untuk Fitur Terpilih
model_cat_final = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0, random_state=42)
model_lgb_final = LGBMClassifier(n_estimators=1000, learning_rate=0.05, max_depth=8, verbose=-1, random_state=42)

ensemble_signifikan = VotingClassifier(
    estimators=[
        ('catboost', model_cat_final),
        ('lightgbm', model_lgb_final)
    ],
    voting='soft'
)

print("\n⏳ Melatih model baru menggunakan variabel pilihan...")
ensemble_signifikan.fit(X_train_new, y_train_new)
print("✅ Pelatihan model final selesai!")

# =========================================================================
# 4. UJI COBA MODEL MENGGUNAKAN DATA TESTING (20%)
# =========================================================================
y_pred_new = ensemble_signifikan.predict(X_test_new)

# Menampilkan Hasil Pengujian
akurasi_baru = accuracy_score(y_test_new, y_pred_new)
print(f"\n🎯 AKURASI MODEL BARU (DENGAN DATA TESTING): {akurasi_baru * 100:.2f}%")

print("\n📋 LAPORAN KLASIFIKASI MODEL BARU:")
print(classification_report(y_test_new, y_pred_new, target_names=le_dict['NObeyesdad'].classes_))

# Menyimpan Model Seleksi Fitur
with open('model_ensemble_signifikan.pkl', 'wb') as f:
    pickle.dump({
        'model': ensemble_signifikan,
        'encoders': le_dict,
        'features': fitur_signifikan,
        'classes': le_dict['NObeyesdad'].classes_.tolist()
    }, f)
print("💾 Model seleksi fitur berhasil disimpan sebagai 'model_ensemble_signifikan.pkl'")
