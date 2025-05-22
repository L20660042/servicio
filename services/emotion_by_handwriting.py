import cv2
import numpy as np
import joblib

model = joblib.load("models/handwriting_emotion_model.joblib")

def extract_handwriting_features(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]

    return np.array([
        len(areas),
        np.mean(areas) if areas else 0,
        np.std(areas) if areas else 0
    ]).reshape(1, -1)

def predict_emotion_from_handwriting(image):
    features = extract_handwriting_features(image)
    emotion = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    emotions = {label: float(score) for label, score in zip(model.classes_, probs)}
    return {"emotions": emotions, "dominant": emotion, "confidence": max(probs)}
