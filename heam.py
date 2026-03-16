import cv2
import numpy as np
import collections
import threading
import speech_recognition as sr
import os
from tensorflow.keras.models import load_model

# --- 1. CONFIGURATION ---
MODEL_PATH = r"C:\Users\My Dell\Downloads\HEAM\emotion_models\best-model.h5"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

NEG_EMOS = ['Angry', 'Disgust', 'Fear', 'Sad']
POS_EMOS = ['Happy', 'Surprise']
NEU_EMOS = ['Neutral']

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

shared_audio_emotion = "Neutral"

# --- 2. AUDIO THREAD ---
def audio_thread():
    global shared_audio_emotion
    recognizer = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                text = recognizer.recognize_google(audio).lower()
                if any(w in text for w in ['bad','hard','sad','no','angry']): shared_audio_emotion = "Sad"
                elif any(w in text for w in ['good','yes','happy','wow']): shared_audio_emotion = "Happy"
                else: shared_audio_emotion = "Neutral"
        except: continue

threading.Thread(target=audio_thread, daemon=True).start()

# --- 3. RESILIENCE ENGINE ---
class UserEngine:
    def __init__(self):
        self.resilience = 100.0
        self.history = collections.deque(maxlen=15)

    def calculate(self, face_emo):
        global shared_audio_emotion
        final_emo = face_emo if shared_audio_emotion == "Neutral" else shared_audio_emotion
        self.history.append(final_emo)
        stable_emo = collections.Counter(self.history).most_common(1)[0][0]

        if stable_emo in NEG_EMOS: self.resilience -= 1.2
        elif stable_emo in POS_EMOS: self.resilience += 1.0
        else: self.resilience += 0.4
        
        self.resilience = np.clip(self.resilience, 0, 100)
        
        if self.resilience < 40: fb = "CRITICAL: High Stress - Intervention required."
        elif stable_emo in NEG_EMOS: fb = "NEGATIVE STATE: Simplifying tasks..."
        elif stable_emo in POS_EMOS: fb = "POSITIVE STATE: User in Flow."
        else: fb = "NEUTRAL: Steady progress."
            
        return int(self.resilience), stable_emo, fb

user_engines = {}
cap = cv2.VideoCapture(0)

# --- 4. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 6, minSize=(40, 40))

    # --- BLACK DASHBOARD & SERIF (TIMES-STYLE) FONT ---
    # FONT_HERSHEY_COMPLEX is the closest match to Times New Roman in OpenCV
    cv2.putText(frame, f"MIC: {shared_audio_emotion}", (20, 40), 
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

    for i, (x, y, w, h) in enumerate(faces):
        try:
            roi = gray_frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            blob = np.reshape(roi.astype('float32') / 255.0, (1, 48, 48, 1))
            preds = model.predict(blob, verbose=0)[0]
            face_emo = EMOTION_LABELS[np.argmax(preds)]

            if i not in user_engines: user_engines[i] = UserEngine()
            res, emo, feedback = user_engines[i].calculate(face_emo)

            # Categorical Colors
            if emo in NEG_EMOS: ui_color = (0, 0, 255)    # Red
            elif emo in POS_EMOS: ui_color = (0, 255, 0) # Green
            else: ui_color = (255, 255, 255)             # White

            # UI Rendering (Using Complex font for all labels)
            cv2.rectangle(frame, (x, y), (x+w, y+h), ui_color, 2)
            cv2.putText(frame, f"{emo} | {res}%", (x, y-35), cv2.FONT_HERSHEY_COMPLEX, 0.7, ui_color, 2)
            
            bar_w = int((w * res) / 100)
            cv2.rectangle(frame, (x, y-20), (x+bar_w, y-15), ui_color, -1)
            cv2.putText(frame, feedback, (x, y+h+30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        except Exception: continue

    cv2.imshow("HUMAN EMOTIONAL ADAPTATION MODEL", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()