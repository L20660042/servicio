from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import time
import logging

from ocr import extract_text
from handwriting_features import calculate_stroke_thickness, calculate_slant_angle, calculate_spacing
from text_emotion import TextEmotionAnalyzer
from fusion import fuse_emotions, dominant_emotion

logging.basicConfig(level=logging.DEBUG)

app = FastAPI(title="Servicio de Análisis de Emociones y Caligrafía")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

text_emotion_analyzer = TextEmotionAnalyzer()

def generate_emotional_advice_ai(emotions: dict, dominant: str) -> str:
    try:
        from transformers import pipeline
        text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")
        
        prompt = (
            f"Emociones detectadas: {emotions}\n"
            f"Emoción dominante: {dominant}\n"
            f"Genera un consejo emocional empático para ayudar a manejar estas emociones:"
        )
        
        output = text_generator(prompt, max_length=60, do_sample=True, temperature=0.7)
        return output[0]["generated_text"].split(":")[-1].strip()

    except Exception as e:
        logging.error(f"Error al generar consejo emocional: {e}")
        return "No se pudo generar un consejo emocional en este momento."

@app.get("/health")
async def health():
    return {"model_loaded": True}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    start_total = time.time()
    try:
        image_bytes = await file.read()

        # OCR (aunque el texto no se usa para el consejo)
        start_ocr = time.time()
        text = extract_text(image_bytes)
        logging.debug(f"OCR time: {time.time() - start_ocr:.2f}s")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        _, img_bin = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Extraer características caligráficas
        start_hw = time.time()
        stroke_thickness = calculate_stroke_thickness(img_bin)
        slant_angle = calculate_slant_angle(img_bin)
        spacing = calculate_spacing(img_bin)
        logging.debug(f"Handwriting features time: {time.time() - start_hw:.2f}s")

        # Emociones a partir del texto (para fusión)
        start_text_emotion = time.time()
        text_emotions = text_emotion_analyzer.analyze(text if text else " ")
        logging.debug(f"Text emotion analysis time: {time.time() - start_text_emotion:.2f}s")

        # Fusión y análisis emocional
        combined_emotions = fuse_emotions(text_emotions, stroke_thickness, slant_angle, spacing)
        dom_emotion = dominant_emotion(combined_emotions)

        # Consejo emocional IA
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, timeout_keep_alive=120)
