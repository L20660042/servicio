import cv2
import numpy as np
import joblib

model = joblib.load("models/sketch_emotion_model.joblib")

def extract_stroke_features(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    features = np.array([
        np.mean(edges),
        np.std(edges),
        np.sum(edges) / edges.size
    ]).reshape(1, -1)
    return features

def predict_emotion_from_strokes(image):
    features = extract_stroke_features(image)
    emotion = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    emotions = {label: float(score) for label, score in zip(model.classes_, probs)}
    return {"emotions": emotions, "dominant": emotion, "confidence": max(probs)}
