from transformers import pipeline
import logging

class TextEmotionAnalyzer:
    def __init__(self):
        self.labels = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]
        self.model_name = "mrm8488/t5-base-finetuned-emotion"
        self.nlp = pipeline("text2text-generation", model=self.model_name, tokenizer=self.model_name)

    def analyze(self, text: str) -> dict:
        text = text[:256]  # Limit input length to 256 chars to reduce inference time

        if not text.strip():
            return {label: 0.0 for label in self.labels}

        input_text = f"emotion: {text}"

        try:
            result = self.nlp(input_text, max_length=10)
            output = result[0]['generated_text'].lower()

            emotion_scores = {label: 0.0 for label in self.labels}
            for label in self.labels:
                if label in output:
                    emotion_scores[label] = 1.0

            if sum(emotion_scores.values()) == 0:
                emotion_scores["neutral"] = 1.0

            total = sum(emotion_scores.values())
            if total > 0:
                emotion_scores = {k: v / total for k, v in emotion_scores.items()}

            return emotion_scores

        except Exception as e:
            logging.error(f"Error in TextEmotionAnalyzer.analyze: {e}")
            return {label: 1.0 if label == "neutral" else 0.0 for label in self.labels}
