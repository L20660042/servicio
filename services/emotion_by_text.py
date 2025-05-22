from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="mrm8488/t5-base-finetuned-emotion", return_all_scores=True)

def predict_emotion_from_text(text: str):
    scores = emotion_classifier(text)[0]
    emotions = {e['label']: e['score'] for e in scores}
    dominant = max(emotions, key=emotions.get)
    return {"emotions": emotions, "dominant": dominant, "confidence": emotions[dominant]}
