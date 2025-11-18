import cv2
import mediapipe as mp
import pickle
import numpy as np

def normalize_landmarks(landmarks_mp):
    """
    Fungsi untuk melakukan normalisasi translasi, skala, dan rotasi.
    """
    # ... (Fungsi ini tidak berubah)
    landmarks = [[lm.x, lm.y] for lm in landmarks_mp.landmark]
    landmarks_np = np.array(landmarks)
    base_point = landmarks_np[0]
    translated_landmarks = landmarks_np - base_point
    max_dist = np.max(np.linalg.norm(translated_landmarks, axis=1))
    if max_dist == 0:
        return translated_landmarks.flatten().tolist()
    scaled_landmarks = translated_landmarks / max_dist
    reference_vector = scaled_landmarks[9] - scaled_landmarks[0]
    angle = np.arctan2(reference_vector[1], reference_vector[0]) - np.pi / 2
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    rotated_landmarks = np.dot(scaled_landmarks, rotation_matrix.T)
    return rotated_landmarks.flatten().tolist()

# --- Buka Model ---
with open('gesture_model_SVC.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Mulai deteksi. Arahkan tanganmu ke kamera. Tekan 'q' untuk keluar.")

WINDOW_NAME = 'Deteksi Gestur Real-Time - Jemmy'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

CONFIDENCE_THRESHOLD = 0.55 

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    
    # vvvvv INI BARIS YANG DIPERBAIKI vvvvv
    image_for_detection = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Seharusnya COLOR_BGR2RGB, bukan COLOR_BGR_RGB
    # ^^^^^ BARIS YANG DIPERBAIKI ^^^^^
    results = hands.process(image_for_detection)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks_for_model = normalize_landmarks(hand_landmarks)
            prediction = model.predict([landmarks_for_model])
            confidence = model.predict_proba([landmarks_for_model])
            predicted_gesture = prediction[0]
            confidence_score = np.max(confidence)
            
            h, w, c = image.shape
            min_x, min_y, max_x, max_y = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                min_x, min_y, max_x, max_y = min(min_x, px), min(min_y, py), max(max_x, px), max(max_y, py)
            
            if confidence_score >= CONFIDENCE_THRESHOLD:
                display_text = f"{predicted_gesture} ({confidence_score:.2f})"
            else:
                display_text = "Mendeteksi..."

            padding = 20
            BOX_COLOR, TEXT_BG_COLOR, TEXT_COLOR = (255, 0, 0), (0, 0, 0), (255, 255, 255)
            
            cv2.rectangle(image, (min_x - padding, min_y - padding), (max_x + padding, max_y + padding), BOX_COLOR, 2)
            
            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x, text_y = min_x - padding, min_y - padding - 10
            
            cv2.rectangle(image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y + 5), TEXT_BG_COLOR, -1)
            cv2.putText(image, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

    # Logika untuk menjaga rasio aspek (tidak berubah)
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
    except:
        break
    if win_w <= 0 or win_h <= 0:
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue
    img_h, img_w, _ = image.shape
    scale = min(win_w / img_w, win_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    x_offset, y_offset = (win_w - new_w) // 2, (win_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    cv2.imshow(WINDOW_NAME, canvas)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()