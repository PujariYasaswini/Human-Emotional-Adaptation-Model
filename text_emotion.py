import torch
from transformers import pipeline
from collections import deque

class TextEmotionModel:
    def __init__(
        self,
        model_name="j-hartmann/emotion-english-distilroberta-base",
        history_size=5,
        min_length=4,
        min_confidence=0.35,
        device=None
    ):
        self.device = device if device is not None else (
            0 if torch.cuda.is_available() else -1
        )

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=self.device
        )

        self.history = deque(maxlen=history_size)
        self.min_length = min_length
        self.min_confidence = min_confidence

        # Normalize labels across modalities
        self.label_map = {
            "joy": "Happy",
            "anger": "Angry",
            "sadness": "Sad",
            "fear": "Fear",
            "surprise": "Surprise",
            "disgust": "Disgust",
            "neutral": "Neutral"
        }

    def _normalize(self, preds):
        normalized = {}
        for p in preds:
            label = self.label_map.get(p["label"], p["label"])
            normalized[label] = normalized.get(label, 0) + p["score"]
        return normalized

    def _smooth(self, dist):
        self.history.append(dist)
        avg = {}

        for d in self.history:
            for k, v in d.items():
                avg[k] = avg.get(k, 0) + v

        return {k: v / len(self.history) for k, v in avg.items()}

    def predict(self, text):
        """
        Input: transcribed text
        Output: emotion + confidence + distribution
        """

        if not text or len(text.strip().split()) < self.min_length:
            return None

        try:
            raw_preds = self.classifier(text)[0]
        except Exception as e:
            print("Text emotion error:", e)
            return None

        normalized = self._normalize(raw_preds)
        smoothed = self._smooth(normalized)

        emotion = max(smoothed, key=smoothed.get)
        confidence = smoothed[emotion]

        if confidence < self.min_confidence:
            return {
                "emotion": "Neutral",
                "confidence": round(confidence, 2),
                "distribution": smoothed
            }

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "distribution": smoothed
        }
