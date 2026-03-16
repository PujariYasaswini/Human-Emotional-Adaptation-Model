# input/audio_stream.py

import sounddevice as sd
import numpy as np

class AudioStream:
    def __init__(self, sr=16000, duration=3):
        self.sr = sr
        self.duration = duration

    def record(self):
        """Records audio and returns numpy array"""
        audio = sd.rec(
            int(self.duration * self.sr),
            samplerate=self.sr,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return audio.flatten()
