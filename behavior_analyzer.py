import numpy as np

class BehaviorAnalyzer:
    def __init__(
        self,
        confidence_threshold=0.6,
        trend_window=10
    ):
        self.confidence_threshold = confidence_threshold
        self.trend_window = trend_window

        self.negative_emotions = {
            "Sad", "Angry", "Nervous", "Confused", "Disgust"
        }
        self.positive_emotions = {
            "Happy", "Excited", "Joy"
        }

    def _negativity_score(self, history):
        score = 0.0
        weight = 0.0

        for h in history:
            if h["confidence"] < self.confidence_threshold:
                continue

            w = h["confidence"]
            weight += w

            if h["emotion"] in self.negative_emotions:
                score += w

        if weight == 0:
            return 0

        return score / weight

    def _emotion_trend(self, history):
        if len(history) < 3:
            return 0

        recent = history[-self.trend_window:]
        trend = []

        for h in recent:
            if h["emotion"] in self.positive_emotions:
                trend.append(1)
            elif h["emotion"] in self.negative_emotions:
                trend.append(-1)
            else:
                trend.append(0)

        return np.mean(trend)

    def _recovery_score(self, history):
        if len(history) < 5:
            return 0

        negatives = [
            h for h in history[-self.trend_window:]
            if h["emotion"] in self.negative_emotions
        ]

        positives = [
            h for h in history[-self.trend_window:]
            if h["emotion"] in self.positive_emotions
        ]

        return len(positives) - len(negatives)
    def analyze(self, emotion_history):
        """
        emotion_history format:
        [
          {"emotion": str, "confidence": float, "timestamp": float},
          ...
        ]
        """

        if not emotion_history:
            return {
                "state": "Unknown",
                "risk": 0.0
            }

        negativity = self._negativity_score(emotion_history)
        trend = self._emotion_trend(emotion_history)
        recovery = self._recovery_score(emotion_history)
        if negativity > 0.65 and trend < -0.3:
            state = "Maladaptive"
        elif negativity > 0.4:
            state = "At-Risk"
        elif trend > 0.2 or recovery > 0:
            state = "Adaptive"
        else:
            state = "Stagnant"

        risk = round(float(negativity * (1 - max(trend, 0))), 2)

        return {
            "state": state,
            "risk": risk,
            "negativity": round(negativity, 2),
            "trend": round(trend, 2),
            "recovery": recovery
        }
