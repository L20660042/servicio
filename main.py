from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import pipeline
import io
import numpy as np
from PIL import Image

app = FastAPI()

# Cargar el modelo de Hugging Face para análisis de emociones
emotion_recognizer = pipeline("image-classification", model="facebook/dino-vit-base-patch16")

# Endpoint para verificar que el modelo está cargado
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Función para analizar la imagen
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen del archivo
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir la imagen a un formato adecuado para el modelo (si es necesario)
        # Aquí puedes hacer preprocesamiento de la imagen si tu modelo lo requiere.
        
        # Usar el modelo para obtener las emociones
        results = emotion_recognizer(image)
        
        # Suponiendo que el modelo retorna una etiqueta de emoción
        emotion = results[0]["label"]  # Esto dependerá de cómo está configurado tu modelo
        
        # Generar el consejo basado en la emoción detectada
        emotional_advice = generate_advice(emotion)
        
        return JSONResponse(content={
            "data": {
                "emotions": {emotion: 1.0},  # Aquí puedes agregar más emociones si el modelo lo permite
                "dominant_emotion": emotion,
                "emotional_advice": emotional_advice
            }
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Función para generar consejos emocionales basados en la emoción detectada
def generate_advice(emotion):
    if emotion == "anger":
        return "Mediante los trazos y el grosor de las líneas, se detecta que estás enojado. Se recomienda salir a distraerte."
    elif emotion == "joy":
        return "Se detecta que estás alegre. ¡Sigue disfrutando este momento positivo!"
    else:
        return "Emoción detectada. Mantén un equilibrio emocional."
