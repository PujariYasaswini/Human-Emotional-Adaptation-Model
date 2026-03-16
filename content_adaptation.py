from openai import OpenAI
import random
import time

class ContentAdaptation:
    def __init__(self, api_key=None, temperature=0.7):
        """
        AI-driven adaptive response generator
        - api_key: OpenAI API key
        - temperature: controls creativity of responses
        """
        self.client = OpenAI(api_key=api_key)
        self.temperature = temperature

    def generate_prompt(self, student_emotion, behavior_state, recent_history, context="lecture"):
        """
        Builds a dynamic prompt for the AI
        """
        history_str = ""
        for h in recent_history[-5:]:
            ts = time.strftime("%H:%M:%S", time.localtime(h.get("timestamp", time.time())))
            history_str += f"[{ts}] Emotion: {h['emotion']}, Confidence: {h['confidence']}\n"

        prompt = (
            f"You are a smart teaching assistant AI. The student currently shows emotion: {student_emotion}, "
            f"behavioral state: {behavior_state}.\n"
            f"Recent emotional history:\n{history_str}\n"
            f"Your task is to respond in a way that:\n"
            f"- is empathetic and supportive\n"
            f"- encourages engagement and understanding\n"
            f"- adapts to the student's emotional state\n"
            f"- is contextually appropriate for a {context} session\n"
            f"- avoids repeating previous responses\n"
            f"- can suggest interactive activities, examples, or feedback prompts\n"
            f"Provide the response in a single paragraph."
        )

        return prompt

    def generate_response(self, student_emotion, behavior_state, recent_history, context="lecture"):
        """
        Returns AI-generated response text
        """
        prompt = self.generate_prompt(student_emotion, behavior_state, recent_history, context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful, empathetic AI teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            ai_text = response.choices[0].message.content.strip()
            return ai_text
        except Exception as e:
            print("Adaptive response generation failed:", e)
            # Fallback: small dynamic suggestion
            fallback_responses = [
                "Let's try an interactive example to make this clearer.",
                "How about a small quiz to check understanding?",
                "Would you like a simpler explanation or example?"
            ]
            return random.choice(fallback_responses)
