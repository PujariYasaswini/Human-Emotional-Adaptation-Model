import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import load_model


class FaceEmotionModel:
    def __init__(self, model_path, history_size=5, face_size=(48, 48)):
        self.model = load_model(model_path)
        self.face_size = face_size

        self.labels = [
            'Angry', 'Disgust', 'Fear',
            'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

        self.history = deque(maxlen=history_size)

        # OpenCV Haar Cascade (FULL FACE)
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # -------------------------
    # Face detection (whole face)
    # -------------------------
    def _detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # Take largest face (covers cheeks + chin)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Expand box to include chin & cheeks
        pad = int(0.15 * h)
        y = max(0, y - pad)
        h = min(frame.shape[0] - y, h + pad)

        return frame[y:y + h, x:x + w]

    # -------------------------
    # CNN preprocessing
    # -------------------------
    def _preprocess_face(self, face_img):
        face = cv2.resize(face_img, self.face_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        return face

    def _smooth_prediction(self, preds):
        self.history.append(preds)
        return np.mean(self.history, axis=0)

    # -------------------------
    # Emotion prediction
    # -------------------------
    def predict(self, frame):
        face = self._detect_face(frame)
        if face is None:
            return None, 0.0

        face_input = self._preprocess_face(face)
        preds = self.model.predict(face_input, verbose=0)[0]
        preds = self._smooth_prediction(preds)

        idx = int(np.argmax(preds))
        return self.labels[idx], float(preds[idx])