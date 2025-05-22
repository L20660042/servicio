from transformers import pipeline

class TextEmotionAnalyzer:
    def __init__(self):
        self.labels = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]
        self.model_name = "mrm8488/t5-base-finetuned-emotion"
        self.nlp = pipeline("text2text-generation", model=self.model_name, tokenizer=self.model_name)

    def analyze(self, text: str) -> dict:
        if not text.strip():
            # Empty text fallback
            return {label: 0.0 for label in self.labels}

        # The model expects a prompt like "emotion: <text>"
        input_text = f"emotion: {text}"

        try:
            result = self.nlp(input_text, max_length=10)
            output = result[0]['generated_text'].lower()

            # Output example: "alegría", "enojo", etc.
            emotion_scores = {label: 0.0 for label in self.labels}
            for label in self.labels:
                if label in output:
                    emotion_scores[label] = 1.0

            # If none matched, assign neutral
            if sum(emotion_scores.values()) == 0:
                emotion_scores["neutral"] = 1.0

            # Normalize (sum=1)
            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v / total for k, v in emotion_scores.items()}

            return emotion_scores

        except Exception:
            # On failure, return neutral
            return {label: 1.0 if label == "neutral" else 0.0 for label in self.labels}
