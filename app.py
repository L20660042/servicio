from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2

from ocr import extract_text
from handwriting_features import calculate_stroke_thickness, calculate_slant_angle, calculate_spacing
from text_emotion import TextEmotionAnalyzer
from fusion import fuse_emotions, dominant_emotion

app = FastAPI(title="Servicio de Análisis de Emociones y Caligrafía")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia por dominios frontend en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

text_emotion_analyzer = TextEmotionAnalyzer()

@app.get("/health")
async def health():
    return {"model_loaded": True}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipo de archivo inválido. Solo JPEG y PNG permitidos.")

    try:
        image_bytes = await file.read()
        text = extract_text(image_bytes)

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_cv is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        _, img_bin = cv2.threshold(img_cv, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        stroke_thickness = calculate_stroke_thickness(img_bin)
        slant_angle = calculate_slant_angle(img_bin)
        spacing = calculate_spacing(img_bin)

        text_emotions = text_emotion_analyzer.analyze(text if text else " ")

        combined_emotions = fuse_emotions(text_emotions, stroke_thickness, slant_angle, spacing)
        dom_emotion = dominant_emotion(combined_emotions)

        return {
            "success": True,
            "data": {
                "text": text,
                "emotions": combined_emotions,
                "dominant_emotion": dom_emotion
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
