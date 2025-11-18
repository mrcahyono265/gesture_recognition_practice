# File 1: konversi_data.py (Versi Optimal)
import os
import json
import csv
import numpy as np

def normalize_landmarks(landmarks):
    """
    Fungsi untuk melakukan normalisasi translasi, skala, dan rotasi.
    """
    landmarks_np = np.array(landmarks)

    # 1. Normalisasi Translasi (Pusat di pergelangan tangan)
    base_point = landmarks_np[0]
    translated_landmarks = landmarks_np - base_point

    # 2. Normalisasi Skala
    # Cari jarak terjauh dari titik pusat (0,0) setelah translasi
    max_dist = np.max(np.linalg.norm(translated_landmarks, axis=1))
    if max_dist == 0: # Hindari pembagian dengan nol
        return translated_landmarks.flatten().tolist()
    
    scaled_landmarks = translated_landmarks / max_dist

    # 3. Normalisasi Rotasi
    # Gunakan vektor dari pergelangan tangan (0) ke pangkal jari tengah (9) sebagai acuan
    reference_vector = scaled_landmarks[9] - scaled_landmarks[0]
    
    # Hitung sudut antara vektor referensi dengan sumbu y (vektor [0, 1])
    # arctan2(y, x) untuk mendapatkan sudut yang benar di semua kuadran
    angle = np.arctan2(reference_vector[1], reference_vector[0]) - np.pi / 2
    
    # Buat matriks rotasi untuk memutar sebesar -angle
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])

    # Terapkan rotasi ke semua landmark
    rotated_landmarks = np.dot(scaled_landmarks, rotation_matrix.T)
    
    return rotated_landmarks.flatten().tolist()

# --- Sisa kode tetap sama ---
annotation_folder = 'dataset/ann_subsample'
output_csv_file = 'dataset/gestures_data.csv'

print(f"Mulai memproses file JSON dari folder: {annotation_folder}")

with open(output_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}']
    writer.writerow(header)

    processed_files_count = 0
    total_gestures_saved = 0
    for filename in os.listdir(annotation_folder):
        if filename.endswith('.json'):
            gesture_label_from_file = filename.split('.')[0]
            
            with open(os.path.join(annotation_folder, filename), 'r') as json_file:
                annotations = json.load(json_file)

                for image_id, annotation in annotations.items():
                    if 'landmarks' in annotation and 'labels' in annotation:
                        for hand_idx, hand_landmarks in enumerate(annotation['landmarks']):
                            if hand_landmarks and len(hand_landmarks) == 21:
                                if hand_idx < len(annotation['labels']):
                                    gesture_label = annotation['labels'][hand_idx]
                                else:
                                    continue

                                if gesture_label == gesture_label_from_file:
                                    # Gunakan fungsi normalisasi yang baru
                                    normalized_flattened_landmarks = normalize_landmarks(hand_landmarks)
                                    
                                    writer.writerow([gesture_label] + normalized_flattened_landmarks)
                                    total_gestures_saved += 1
            
            processed_files_count += 1
            print(f"Selesai memproses: {filename}")

print(f"\nSelesai! {processed_files_count} file gestur berhasil diproses.")
print(f"Total {total_gestures_saved} data gestur valid berhasil disimpan di '{output_csv_file}'")