import easyocr
import numpy as np

reader = easyocr.Reader(['es', 'en'], gpu=False)

def extract_text(image):
    img_array = np.array(image)
    result = reader.readtext(img_array)
    return " ".join([text for (_, text, _) in result])
