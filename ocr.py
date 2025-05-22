import pytesseract
from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen inválida")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return processed

def extract_text(image_bytes: bytes) -> str:
    processed_img = preprocess_image(image_bytes)
    pil_img = Image.fromarray(processed_img)

    custom_config = r'--oem 1 --psm 6 -l spa'
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    return text.strip()
