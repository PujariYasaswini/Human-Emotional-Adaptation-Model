# emotion_models/voice_emotion_model.py
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model

class VoiceEmotionModel:
    def __init__(self, model_path):
        # Load pre-trained voice emotion model
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        self.labels = ['Angry', 'Calm', 'Fearful', 'Happy', 'Sad', 'Nervous']

    def record_audio(self, duration=3, fs=16000):
        print("🎤 Recording audio...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        return audio

    def predict(self, audio):
        # Placeholder prediction
        # TODO: replace with real preprocessing + model prediction
        # For now, return random label and confidence
        emotion = np.random.choice(self.labels)
        confidence = np.random.uniform(0.6, 1.0)
        return emotion, confidence
