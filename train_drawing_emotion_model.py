import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.model_selection import train_test_split

def load_dataset(path='dataset_drawing'):
    X, y = [], []
    for emotion in os.listdir(path):
        for img_file in os.listdir(f"{path}/{emotion}"):
            img = cv2.imread(f"{path}/{emotion}/{img_file}", cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, 50, 150)
            features = [
                np.mean(edges),
                np.std(edges),
                np.sum(edges) / edges.size
            ]
            X.append(features)
            y.append(emotion)
    return np.array(X), np.array(y)

X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = MLPClassifier(hidden_layer_sizes=(30, 10), max_iter=500)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))
joblib.dump(clf, "models/sketch_emotion_model.joblib")
