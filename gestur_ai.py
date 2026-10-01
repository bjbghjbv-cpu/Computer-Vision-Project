import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os
from gtts import gTTS
import pygame


model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Mengunduh model MediaPipe... Mohon tunggu.")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Berhasil mengunduh model.")


PHRASES = {
    "hi": "hi",
    "my": "my",
    "name": "name",
    "is": "is",
    "bernard": "bernard",
    "thankyou": "thank you",
    "nice to meet you": "nice to meet you",
}

AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

def audio_path(key):
    return os.path.join(AUDIO_DIR, key.replace(" ", "_") + ".mp3")

print("Menyiapkan file suara (butuh internet hanya untuk pertama kali)...")
for key, spoken in PHRASES.items():
    path = audio_path(key)
    if not os.path.exists(path):
        try:
            gTTS(text=spoken, lang='en', slow=False).save(path)
            print(f"Dibuat: {path}")
        except Exception as e:
            print(f"Gagal membuat suara untuk {key}: {e}")

pygame.mixer.init()

def speak(key):
    path = audio_path(key)
    if not os.path.exists(path):
        return False
    if pygame.mixer.music.get_busy():
        return False
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        return True
    except Exception as e:
        print(f"Error memutar suara: {e}")
        return False


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def get_fingers(lm):
    fingers = []
    # Ibu jari
    fingers.append(dist(lm[4], lm[17]) > dist(lm[3], lm[17]))
    # 4 Jari lainnya
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    for tip, pip in zip(tips, pips):
        fingers.append(dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]))
    return fingers


def recognize(hand_info):
    text = ""
    num_hands = len(hand_info)
    
    if num_hands == 2:
        f1, f2 = hand_info[0]['fingers'], hand_info[1]['fingers']
        if all(f1) and all(f2):
            text = "hi"
        elif f1[0] and not any(f1[1:]) and f2[0] and not any(f2[1:]):
            text = "my"
            
    elif num_hands == 1:
        f = hand_info[0]['fingers']
        lm = hand_info[0]['landmarks']
        
        if not any(f[1:]):
            text = "name"
        elif f[1] and not any(f[2:]):
            text = "bernard"
        elif f[1] and f[2] and f[3] and f[4]:
            text = "thankyou"
        elif f[1] and f[2] and not f[3] and not f[4]:
           
            if lm[8].y > lm[0].y - 0.2: 
                text = "is" 
            else:
                text = "nice to meet you"  
                
    return text


cap = cv2.VideoCapture(0)
last_gesture_time = 0.0
gesture_cooldown = 2.0
STABLE_FRAMES = 5
candidate = ""
candidate_count = 0

print("Sistem deteksi gestur berjalan. Tekan q untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    current_text = ""
    hand_info = []
    
    if detection_result.hand_landmarks:
        for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), cv2.FILLED)
                
            fingers = get_fingers(hand_landmarks)
            hand_info.append({'landmarks': hand_landmarks, 'fingers': fingers})
            
            status = "".join("1" if x else "0" for x in fingers)
            cv2.putText(frame, f"Tangan {idx + 1}: {status}", (30, h - 30 - idx * 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
        current_text = recognize(hand_info)
        
    if current_text and current_text == candidate:
        candidate_count += 1
    else:
        candidate = current_text
        candidate_count = 1 if current_text else 0
        
    if candidate_count >= STABLE_FRAMES and (time.time() - last_gesture_time > gesture_cooldown):
        if speak(candidate):
            last_gesture_time = time.time()
            
    if current_text:
        cv2.putText(frame, f"Deteksi: {current_text.upper()}", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
    cv2.imshow("Gestur AI Tangan", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
