# latih_model.py - Versi MLPClassifier (Neural Network)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier # <- Ganti modelnya di sini
from sklearn.metrics import accuracy_score
import pickle

# 1. Muat Dataset
df = pd.read_csv('dataset/gestures_data.csv')

X = df.drop('label', axis=1)
y = df['label']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Inisialisasi Model
# hidden_layer_sizes: (100, 50) berarti ada 2 lapisan tersembunyi, satu dengan 100 neuron, satu lagi 50 neuron.
# max_iter: Jumlah maksimum iterasi pelatihan.
# alpha: Parameter regularisasi untuk menghindari overfitting.
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=1e-4,
                      solver='adam', random_state=42, verbose=False)

# 4. Latih Model
print("Mulai melatih model Jaringan Saraf Tiruan (MLP)...")
model.fit(X_train, y_train)
print("Model selesai dilatih.")

# 5. Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

# 6. Simpan Model
with open('gesture_model_MLP.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai 'gesture_model.pkl'")