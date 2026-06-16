# Real-Time Hand Gesture Recognition

---

## 📖 Overview

Sistem ini membaca *input* dari kamera, mendeteksi kerangka tangan (21 titik *landmark*), menormalisasi titik-titik tersebut (berdasarkan translasi, skala, dan rotasi), lalu memprediksi gestur apa yang sedang dilakukan menggunakan model *Machine Learning* klasik yang telah dilatih sebelumnya. Proyek ini sangat ringan dan dapat berjalan lancar di CPU tanpa memerlukan GPU khusus.

---

## 🎯 Objectives

- Mengonversi data anotasi gestur mentah (JSON) menjadi dataset berformat tabular (CSV).
- Menerapkan prapemrosesan (*normalization*) matematika pada koordinat tangan agar model kebal terhadap perubahan posisi, jarak, dan kemiringan tangan ke kamera.
- Membandingkan dan menyediakan berbagai opsi model klasifikasi (Random Forest, SVM, Multi-Layer Perceptron, Gradient Boosting).
- Menyediakan antarmuka *real-time* menggunakan OpenCV untuk menampilkan hasil deteksi secara langsung.

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV (`cv2`), MediaPipe (`mp.solutions.hands`)
- **Machine Learning:** Scikit-Learn (`SVC`, `RandomForestClassifier`, `MLPClassifier`, `GradientBoostingClassifier`)
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Pickle

---

## ✨ Key Features

- **Robust Normalization:** Memiliki algoritma normalisasi khusus (`normalize_landmarks`) yang menggeser titik pusat ke pergelangan tangan, menskalakan ukuran, dan memutar koordinat agar sejajar. Ini membuat prediksi jauh lebih akurat.
- **Multiple Algorithm Options:** Tersedia 4 *script* pelatihan terpisah (GBC, MLP, RFC, SVC) sehingga Anda bisa bereksperimen mencari model dengan akurasi terbaik.
- **Real-Time Bounding Box & Confidence Score:** Menampilkan kotak pendeteksi di sekitar tangan beserta label prediksi dan tingkat persentase keyakinan (*confidence score*).
- **Auto-Scaling Window:** Jendela OpenCV dirancang untuk mempertahankan rasio aspek secara otomatis saat di- *resize*.

---

## 📁 Project Structure

```bash
├── dataset/
│   ├── ann_subsample/        # Folder berisi file anotasi JSON (input mentah)
│   └── gestures_data.csv     # File hasil ekstraksi (dibuat otomatis oleh script)
├── konversi_data.py          # Script untuk memproses JSON menjadi CSV + normalisasi
├── latih_model_GBC.py        # Script training model Gradient Boosting
├── latih_model_MLP.py        # Script training model Neural Network (MLP)
├── latih_model_RFC.py        # Script training model Random Forest
├── latih_model_SVC.py        # Script training model Support Vector Machine
└── run.py                    # Script utama untuk menjalankan deteksi real-time via webcam
```

---

## 🚀 Installation & Setup

### 0. Requirements

Pastikan Anda menggunakan Python 3.8, 3.9, atau 3.10.

### 1. Clone repository

```bash
git clone https://github.com/username/gesture_recognition_practice.git
cd gesture_recognition_practice
```

### 2. Buat Virtual Environment & Install Dependencies

Sangat disarankan menggunakan virtual environment agar library tidak bentrok.

```bash
# Membuat virtual environment
python -m venv env

# Aktivasi environment (Windows)
env\Scripts\activate
# Aktivasi environment (Mac/Linux)
source env/bin/activate

# Install library yang dibutuhkan
pip install opencv-python mediapipe scikit-learn pandas numpy
```

---

## 💻 Cara Penggunaan

Proyek ini memiliki alur kerja 3 tahap: Ekstraksi Data -> Pelatihan Model -> Inferensi Real-Time.

### Langkah 1: Siapkan Dataset

Pastikan data anotasi JSON Anda berada di dalam folder dataset/ann_subsample/. Lalu jalankan perintah berikut untuk mengekstrak titik landmark dan menyimpannya ke dataset/gestures_data.csv:

```
python konversi_data.py
```

### Langkah 2: Latih Model

Pilih salah satu script pelatihan untuk melatih model klasifikasi. Misalnya, untuk melatih menggunakan algoritma Support Vector Machine (SVC):

```
python latih_model_SVC.py
```

Script ini akan menghasilkan file bobot model (misal: gesture_model_SVC.pkl).

### Langkah 3: Jalankan Deteksi Real-Time

Pastikan nama file .pkl yang dimuat di dalam run.py sudah sesuai dengan model yang Anda latih di Langkah 2. Lalu jalankan:

```
python run.py
```

Arahkan tangan Anda ke kamera. Tekan tombol 'q' pada keyboard untuk menutup aplikasi.

---

## 🧑‍💻 Author

Mohammad Ridho Cahyono

Full Stack Developer | Leadership Experience in Technology & Innovation

Developing Digital Solutions Through Web Development, Machine Learning, and IoT to Help Businesses and Organizations Grow.