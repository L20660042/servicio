def fuse_emotions(text_emotions: dict, stroke_thickness: float, slant_angle: float, spacing: float) -> dict:
    handwriting_emotions = {
        "enojo": min(max(stroke_thickness / 5.0, 0), 1),
        "tristeza": min(max(-slant_angle / 45.0, 0), 1),
        "alegría": min(max(slant_angle / 45.0, 0), 1),
        "neutral": min(max(spacing / 10.0, 0), 1),
        "asco": 0.0,
        "miedo": 0.0,
        "sorpresa": 0.0
    }

    total = sum(handwriting_emotions.values())
    if total > 0:
        handwriting_emotions = {k: v / total for k, v in handwriting_emotions.items()}

    combined = {}
    for emotion in text_emotions:
        combined[emotion] = 0.6 * text_emotions.get(emotion, 0) + 0.4 * handwriting_emotions.get(emotion, 0)

    total_combined = sum(combined.values())
    if total_combined > 0:
        combined = {k: v / total_combined for k, v in combined.items()}

    return combined

def dominant_emotion(emotions: dict) -> str:
    return max(emotions, key=emotions.get)
