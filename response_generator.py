import random

class ResponseGenerator:
    def generate(self, emotion, strategy, history, context=None):
        """
        emotion: str, current emotion
        strategy: str, selected strategy
        history: list of past emotions
        context: dict, optional, e.g., {"topic": "algebra", "difficulty": "medium"}
        """
        context = context or {}
        topic = context.get("topic", "this topic")
        difficulty = context.get("difficulty", "medium")

        if strategy == "RETEACH_WITH_EXAMPLE":
            options = [
                f"Let’s pause for a moment. I’ll explain {topic} again using a simpler example.",
                f"Let’s revisit {topic} with another example to make it clearer."
            ]
            return random.choice(options)

        if strategy == "ASK_DIAGNOSTIC_QUESTION":
            options = [
                f"Before moving ahead, can you tell me which part of {topic} feels unclear?",
                f"Which part of {topic} is confusing you? Let’s clarify it."
            ]
            return random.choice(options)

        if strategy == "CHANGE_ACTIVITY":
            options = [
                f"We’ve been discussing {topic} for a while — let’s switch gears with a quick activity.",
                f"Let's try something different to refresh our focus from {topic}."
            ]
            return random.choice(options)

        if strategy == "INCREASE_INTERACTION":
            options = [
                f"Let’s make this interactive. Try answering this based on {topic}.",
                f"Can you explain {topic} in your own words?"
            ]
            return random.choice(options)

        if strategy == "EMOTIONAL_REASSURANCE":
            options = [
                f"It’s okay if this feels challenging. {topic} is tricky, and we’ll take it step by step.",
                f"Don’t worry — {topic} can be tough. We’ll go slowly together.",
                f"Take a deep breath. {topic} is tricky, but we can handle it."
            ]
            return random.choice(options)

        if strategy == "INCREASE_DIFFICULTY":
            options = [
                f"You’re doing great. Let’s try a slightly more advanced idea related to {topic}.",
                f"Now that you’re comfortable, let’s challenge ourselves a bit more with {topic}."
            ]
            return random.choice(options)

        return "Let me know how you’re feeling right now."
