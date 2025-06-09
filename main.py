from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import pytesseract
import io
import logging

app = FastAPI()

# Configurar CORS para permitir solicitudes de cualquier dominio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Cargar el modelo de Hugging Face para el análisis de emociones
emotion_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
emotion_labels = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]

# Función para analizar las emociones en el texto extraído
def analyze_text_emotion(text):
    try:
        result = emotion_model(text, candidate_labels=emotion_labels)
        dominant_emotion = result['labels'][0]
        emotions = {emotion: result['scores'][i] for i, emotion in enumerate(result['labels'])}

        # Generar un consejo basado en la emoción dominante
        if dominant_emotion == "enojo":
            advice = "Parece que estás enojado. Tal vez tomar un descanso y relajarte te ayude."
        elif dominant_emotion == "tristeza":
            advice = "Parece que estás triste. Hablar con alguien podría ayudarte a sentirte mejor."
        elif dominant_emotion == "alegría":
            advice = "¡Estás feliz! Disfruta este momento positivo."
        elif dominant_emotion == "miedo":
            advice = "Parece que tienes miedo. Respira profundamente y tómate un momento para calmarte."
        elif dominant_emotion == "sorpresa":
            advice = "¡Qué sorpresa! Aprovecha esta emoción para explorar nuevas oportunidades."
        elif dominant_emotion == "neutral":
            advice = "Estás en un estado equilibrado. Mantén esta calma."
        elif dominant_emotion == "asco":
            advice = "Estás disgustado. Tómate un tiempo para reflexionar sobre lo que sientes."

        return emotions, dominant_emotion, advice

    except Exception as e:
        logging.error(f"Error al analizar el texto: {e}")
        return {"error": str(e)}

# Endpoint de salud del servicio
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Endpoint para analizar la imagen
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen cargada
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Usar Tesseract para extraer texto de la imagen
        text = pytesseract.image_to_string(image)

        # Si no se encuentra texto, devolver un error
        if not text.strip():
            return {"error": "No se pudo extraer texto de la imagen"}

        # Analizar las emociones del texto extraído
        emotions, dominant_emotion, advice = analyze_text_emotion(text)

        return {
            "data": {
                "text": text,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "emotional_advice": advice,
            }
        }

    except Exception as e:
        logging.error(f"Error al procesar la imagen: {e}")
        return {"error": str(e)}
