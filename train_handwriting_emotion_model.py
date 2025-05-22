import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.model_selection import train_test_split

def load_dataset(path='dataset_handwriting'):
    X, y = [], []
    for emotion in os.listdir(path):
        for img_file in os.listdir(f"{path}/{emotion}"):
            img = cv2.imread(f"{path}/{emotion}/{img_file}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
            features = [
                len(areas),
                np.mean(areas) if areas else 0,
                np.std(areas) if areas else 0
            ]
            X.append(features)
            y.append(emotion)
    return np.array(X), np.array(y)

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "models/handwriting_emotion_model.joblib")
