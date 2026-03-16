class StrategyEngine:
    def decide_strategy(
        self,
        emotion,
        state=None,
        confidence=0.5,
        trend="stable",
        context=None
    ):
        """
        emotion   : str
        state     : output of BehaviorAnalyzer (optional)
        confidence: float [0–1]
        trend     : 'increasing' | 'decreasing' | 'stable'
        context   : dict with session info
        """

        if context is None:
            context = {}

        lecture_time = context.get("lecture_time", 0)

        if emotion == "Confused" and confidence > 0.7:
            if trend == "increasing":
                return "RETEACH_WITH_EXAMPLE"
            return "ASK_DIAGNOSTIC_QUESTION"

        if emotion == "Bored":
            if lecture_time > 30:
                return "CHANGE_ACTIVITY"
            return "INCREASE_INTERACTION"

        if emotion in {"Frustrated", "Angry", "Sad"}:
            return "EMOTIONAL_REASSURANCE"

        if emotion == "Happy":
            return "INCREASE_DIFFICULTY"

        if state == "Maladaptive":
            return "CHECK_IN_GENTLY"

        return "CHECK_IN"
