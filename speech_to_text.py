import speech_recognition as sr


class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen(self, timeout=3):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout)

            text = self.recognizer.recognize_google(audio)
            return text

        except Exception:
            return ""
