from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pytesseract
from transformers import pipeline
import io

# Inicializamos la aplicación FastAPI
app = FastAPI()

# Cargamos el pipeline para análisis de emociones (puedes reemplazar este modelo con el que prefieras)
emotion_analysis_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Ruta de salud para verificar si el servicio está activo
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Función para analizar la imagen
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Leemos la imagen cargada por el usuario
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Extraemos el texto de la imagen usando OCR (Tesseract)
        text = pytesseract.image_to_string(image)

        # Analizamos las emociones en el texto extraído
        analysis = emotion_analysis_pipeline(text)
        emotions = {emotion['label']: emotion['score'] for emotion in analysis}
        dominant_emotion = max(emotions, key=emotions.get)

        # Generamos el consejo emocional basado en la emoción dominante
        advice = generar_consejo_emocional(dominant_emotion)

        # Devolvemos los resultados
        return {"data": {
            "text": text,
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "emotional_advice": advice
        }}
    except Exception as e:
        return {"error": str(e)}

# Función para generar el consejo emocional
def generar_consejo_emocional(emocion):
    consejos = {
        "anger": "Respira profundamente y relájate. Considera salir a caminar para liberar el estrés.",
        "joy": "¡Sigue con ese excelente estado de ánimo! Sigue haciendo cosas que te hagan feliz.",
        "sadness": "Está bien sentirse triste, pero hablar con alguien podría ayudarte.",
        "fear": "Prueba con ejercicios de relajación y enfócate en tu respiración.",
        "surprise": "¡Vaya! Es algo inesperado. Tómate tu tiempo para procesarlo.",
        "disgust": "Tómate un tiempo para ti mismo, relájate y despeja tu mente.",
        "neutral": "Parece que estás tranquilo. Disfruta el momento presente."
    }
    return consejos.get(emocion, "Mantente positivo/a!")
