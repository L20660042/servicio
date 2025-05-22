import cv2
import numpy as np

def calculate_stroke_thickness(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
    thickness = np.mean(dist_transform[dist_transform > 0])
    return float(thickness)

def calculate_slant_angle(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    avg_angle = np.mean(angles)
    return float(avg_angle)

def calculate_spacing(image: np.ndarray) -> float:
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    horizontal_proj = np.sum(binary, axis=0)
    gaps = np.sum(horizontal_proj == 0)
    return float(gaps)
