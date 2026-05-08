#=====================
#train py
#=====================
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Load Data
path = r"C:\Users\ASUS\OneDrive\Documents\SEMESTER 6\Mechine Learning\D2 (Uni)\KEL.2 obesitas Projek MCL 2.xlsx"
df = pd.read_excel(path, sheet_name=0)

# Preprocessing
le_dict = {}
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

# SMOTE & Split
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Train
model = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=8, verbose=0)
model.fit(X_train, y_train)

# Simpan Model & Metadata
with open('model_final.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'encoders': le_dict,
        'features': X.columns.tolist(),
        'classes': le_dict['NObeyesdad'].classes_.tolist()
    }, f)

print("✅ Model Berhasil Disimpan sebagai 'model_final.pkl'")