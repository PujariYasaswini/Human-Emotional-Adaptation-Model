import numpy as np
from collections import deque

class ConfidenceEstimator:
    def __init__(
        self,
        history_size=5,
        entropy_weight=0.4,
        agreement_weight=0.35,
        stability_weight=0.25
    ):
        """
        Weights define importance of each confidence factor
        """
        self.entropy_weight = entropy_weight
        self.agreement_weight = agreement_weight
        self.stability_weight = stability_weight

        self.emotion_history = deque(maxlen=history_size)

    def _entropy_confidence(self, distribution):
        """
        Lower entropy → higher confidence
        """
        if not isinstance(distribution, dict) or not distribution:
            return 0.5  # neutral confidence if unavailable

        probs = np.array(list(distribution.values()), dtype=np.float32)

        # Remove NaNs or negative values
        probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)

        total = probs.sum()
        if total <= 0:
            return 0.5  # avoid division by zero

        probs = probs / total
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0  # avoid log(1)=0
        confidence = 1.0 - (entropy / max_entropy)
        if np.isnan(confidence) or np.isinf(confidence):
            confidence = 0.5
        return float(np.clip(confidence, 0.0, 1.0))
    def _agreement_confidence(self, face, voice, text):
        emotions = []
        for m in (face, voice, text):
            if isinstance(m, dict) and "emotion" in m:
                emotions.append(m["emotion"])
        if len(emotions) <= 1:
            return 0.5  # not enough info
        most_common = max(set(emotions), key=emotions.count)
        confidence = emotions.count(most_common) / len(emotions)

        return float(np.clip(confidence, 0.0, 1.0))
    def _stability_confidence(self, emotion):
        self.emotion_history.append(emotion)

        if len(self.emotion_history) < 2:
            return 0.5  

        confidence = sum(e == emotion for e in self.emotion_history) / len(self.emotion_history)

        return float(np.clip(confidence, 0.0, 1.0))

    def estimate(
        self,
        fused_result,
        face=None,
        voice=None,
        text=None
    ):
        """
        fused_result must contain at least:
        {
          "emotion": str
        }
        'distribution' is optional:
        {
          "emotion1": prob1,
          "emotion2": prob2,
          ...
        }
        """

        emotion = fused_result.get("emotion")
        distribution = fused_result.get("distribution")

        entropy_conf = self._entropy_confidence(distribution)
        agreement_conf = self._agreement_confidence(face, voice, text)
        stability_conf = self._stability_confidence(emotion)

        final_confidence = (
            entropy_conf * self.entropy_weight +
            agreement_conf * self.agreement_weight +
            stability_conf * self.stability_weight
        )

        if np.isnan(final_confidence) or np.isinf(final_confidence):
            final_confidence = 0.5

        return round(float(np.clip(final_confidence, 0.0, 1.0)), 2)
