# latih_model.py - Versi SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # <- Ganti modelnya di sini
from sklearn.metrics import accuracy_score
import pickle

# 1. Muat Dataset
df = pd.read_csv('dataset/gestures_data.csv')

X = df.drop('label', axis=1)
y = df['label']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Inisialisasi Model
# Gunakan SVC. `probability=True` agar kita bisa mendapatkan skor kepercayaan nanti
# Kernel 'rbf' adalah pilihan default yang bagus untuk banyak kasus
model = SVC(kernel='rbf', probability=True, random_state=42)

# 4. Latih Model
print("Mulai melatih model SVC... (Mungkin butuh waktu lebih lama)")
model.fit(X_train, y_train)
print("Model selesai dilatih.")

# 5. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# 6. Simpan Model
with open('gesture_model_SVC.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai 'gesture_model.pkl'")