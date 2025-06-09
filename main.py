from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import io
import logging

# Crear la aplicación FastAPI
app = FastAPI()

# Configuración de CORS: Permitir todos los orígenes y todos los métodos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Carga el modelo de Hugging Face para el análisis de emociones
emotion_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Emociones disponibles para el análisis
emotion_labels = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]

# Función para analizar imagen y devolver emociones
def analyze_image_emotion(image_bytes):
    try:
        # Cargar la imagen
        image = Image.open(io.BytesIO(image_bytes))
        # Aquí puedes procesar la imagen con algún modelo de detección de emociones visuales
        # Por ejemplo, utilizando una API o un modelo preentrenado para clasificación de emociones de la imagen

        # Este es un ejemplo genérico de análisis de texto, puede ser modificado para imágenes
        result = emotion_model("Analiza el rostro de la persona", candidate_labels=emotion_labels)
        dominant_emotion = result['labels'][0]
        emotions = {emotion: result['scores'][i] for i, emotion in enumerate(result['labels'])}
        
        # Generar consejo emocional
        advice = ""
        if dominant_emotion == "enojo":
            advice = "Mediante los trazos y el grosor de las líneas, se detecta que estás enojado. Se recomienda salir a distraerte."
        elif dominant_emotion == "tristeza":
            advice = "Parece que estás triste, tal vez tomar un descanso y hablar con alguien podría ayudarte."

        return emotions, dominant_emotion, advice

    except Exception as e:
        logging.error(f"Error al analizar la imagen: {e}")
        return {"error": str(e)}

# Endpoint de salud del servicio
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Endpoint para analizar la imagen
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        emotions, dominant_emotion, advice = analyze_image_emotion(image_bytes)

        return {
            "data": {
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "emotional_advice": advice,
            }
        }

    except Exception as e:
        logging.error(f"Error al procesar la imagen: {e}")
        return {"error": str(e)}
