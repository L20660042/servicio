import time
import logging
import numpy as np
import pytesseract
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Inicializar FastAPI
app = FastAPI()

# Modelo de emociones de HuggingFace
text_emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Función OCR
def extract_text(image_bytes):
    # Usar pytesseract para convertir imagen a texto
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    text = pytesseract.image_to_string(img)
    return text

# Función de análisis de características de la caligrafía
def calculate_stroke_thickness(img_bin):
    # Placeholder para calcular grosor de trazo
    return 1.0

def calculate_slant_angle(img_bin):
    # Placeholder para calcular ángulo de inclinación
    return 0.0

def calculate_spacing(img_bin):
    # Placeholder para calcular espaciado
    return 1.0

# Fusión de emociones
def fuse_emotions(text_emotions, stroke_thickness, slant_angle, spacing):
    combined_emotions = text_emotions.copy()
    # Agregar análisis manuscrito a la emoción del texto (simplificado)
    return combined_emotions

# Determinar la emoción dominante
def dominant_emotion(emotions):
    return max(emotions, key=emotions.get)

# Generar recomendaciones emocionales
def generate_emotional_advice_ai(emotions, dom_emotion):
    advice = f"Recomendación basada en la emoción dominante: {dom_emotion}"
    return advice

# Ruta de salud
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Ruta para analizar la imagen
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    start_total = time.time()
    try:
        image_bytes = await file.read()

        # OCR
        start_ocr = time.time()
        text = extract_text(image_bytes)
        logging.debug(f"OCR time: {time.time() - start_ocr:.2f}s")

        # Decodificar imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        # Binarizar
        _, img_bin = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Caligrafía
        start_hw = time.time()
        stroke_thickness = calculate_stroke_thickness(img_bin)
        slant_angle = calculate_slant_angle(img_bin)
        spacing = calculate_spacing(img_bin)
        logging.debug(f"Handwriting features time: {time.time() - start_hw:.2f}s")

        # Emociones desde texto
        start_text_emotion = time.time()
        text_emotions = text_emotion_analyzer(text if text else " ")
        logging.debug(f"Text emotion analysis time: {time.time() - start_text_emotion:.2f}s")

        # Fusión de emociones
        combined_emotions = fuse_emotions(text_emotions, stroke_thickness, slant_angle, spacing)
        dom_emotion = dominant_emotion(combined_emotions)

        # Consejos emocionales
        emotional_advice = generate_emotional_advice_ai(combined_emotions, dom_emotion)

        logging.debug(f"Total analyze time: {time.time() - start_total:.2f}s")

        return {
            "success": True,
            "data": {
                "emotions": combined_emotions,
                "dominant_emotion": dom_emotion,
                "emotional_advice": emotional_advice
            }
        }

    except Exception as e:
        logging.error(f"Error in analyze-image: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
