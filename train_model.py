import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

# =====================
# BACA DATASET
# =====================

df = pd.read_csv("social_media.csv")

# =====================
# BUAT TARGET
# =====================

def kategori(score):
    if score <= 3:
        return "Rendah"
    elif score <= 6:
        return "Sedang"
    else:
        return "Tinggi"

df["Addiction_Level"] = df["Addicted_Score"].apply(kategori)

print("\nDistribusi Awal:")
print(df["Addiction_Level"].value_counts())

# =====================
# PILIH FITUR
# =====================

selected_columns = [
    "Age",
    "Gender",
    "Academic_Level",
    "Avg_Daily_Usage_Hours",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Relationship_Status",
    "Conflicts_Over_Social_Media",
    "Addiction_Level"
]

df = df[selected_columns]

# =====================
# ENCODING
# =====================

encoders = {}

for col in [
    "Gender",
    "Academic_Level",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Relationship_Status",
    "Addiction_Level"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# =====================
# X DAN Y
# =====================

X = df.drop("Addiction_Level", axis=1)
y = df["Addiction_Level"]

# =====================
# BALANCING DATA
# =====================

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nDistribusi Setelah SMOTE:")
print(pd.Series(y_resampled).value_counts())

# =====================
# SPLIT DATA
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

# =====================
# MODEL KNN
# =====================

knn = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance"
)

knn.fit(X_train, y_train)

# =====================
# EVALUASI
# =====================

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAkurasi:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================
# SIMPAN
# =====================

joblib.dump(knn, "model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("\nModel berhasil disimpan!")