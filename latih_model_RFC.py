import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('dataset/gestures_data.csv')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Mulai melatih model...")
model.fit(X_train, y_train)
print("Model selesai dilatih.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi model: {accuracy * 100:.2f}%")

with open('gesture_model_RFC.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan sebagai 'gesture_model.pkl'")