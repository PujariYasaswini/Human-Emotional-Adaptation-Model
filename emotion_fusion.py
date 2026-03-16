import numpy as np
from collections import deque

class EmotionFusion:
    def __init__(self, history_size=5, weights=None):
        self.weights = weights or {
            "face": 0.4,
            "voice": 0.35,
            "text": 0.25
        }

        self.history = deque(maxlen=history_size)

        self.positive = {"Happy", "Joy", "Excited"}
        self.negative = {"Sad", "Angry", "Fear", "Disgust", "Nervous", "Confused"}
        self.neutral = {"Neutral", "Calm", "Bored"}

    def _normalize(self, dist):
        total = sum(dist.values())
        if total == 0:
            return dist
        return {k: v / total for k, v in dist.items()}

    def _merge(self, inputs):
        combined = {}

        for modality, result in inputs.items():
            if not result:
                continue

            weight = self.weights.get(modality, 0)
            confidence = result.get("confidence", 1.0) or 0.0

            dist = result.get("distribution")
            if not isinstance(dist, dict):
                emotion = result.get("emotion")
                if not emotion:
                    continue
                dist = {emotion: 1.0}

            for emotion, score in dist.items():
                combined[emotion] = combined.get(emotion, 0) + (
                    score * weight * confidence
                )

        return self._normalize(combined)

    def _smooth(self, dist):
        self.history.append(dist)

        avg = {}
        for d in self.history:
            for k, v in d.items():
                avg[k] = avg.get(k, 0) + v

        return self._normalize(avg)

    def _category(self, emotion):
        if emotion in self.positive:
            return "Positive"
        if emotion in self.negative:
            return "Negative"
        return "Neutral"
    def fuse(self, face=None, voice=None, text=None):
        """
        Inputs can be SIMPLE:
        {
          "emotion": str,
          "confidence": float
        }

        OR FULL:
        {
          "emotion": str,
          "confidence": float,
          "distribution": dict
        }
        """

        merged = self._merge({
            "face": face,
            "voice": voice,
            "text": text
        })

        if not merged:
            return {
                "emotion": "Neutral",
                "category": "Neutral",
                "confidence": 0.0,
                "distribution": {}
            }

        smoothed = self._smooth(merged)

        final_emotion = max(smoothed, key=smoothed.get)
        confidence = float(smoothed[final_emotion])
        category = self._category(final_emotion)

        return {
            "emotion": final_emotion,
            "category": category,
            "confidence": confidence,
            "distribution": {final_emotion: confidence}
        }
