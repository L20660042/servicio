from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services.ocr import extract_text
from services.emotion_by_text import predict_emotion_from_text
from services.emotion_by_drawing import predict_emotion_from_strokes
from services.emotion_by_handwriting import predict_emotion_from_handwriting
from utils.image_utils import read_image_from_upload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    image = await read_image_from_upload(file)
    text = extract_text(image)

    if text.strip():
        result = predict_emotion_from_text(text)
    else:
        handwriting_result = predict_emotion_from_handwriting(image)
        drawing_result = predict_emotion_from_strokes(image)

        # Si handwriting es confiable, priorizarlo
        result = handwriting_result if handwriting_result["confidence"] > 0.5 else drawing_result

    return {
        "success": True,
        "data": {
            "text": text,
            "emotions": result["emotions"],
            "dominant_emotion": result["dominant"]
        }
    }
