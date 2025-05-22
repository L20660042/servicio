from PIL import Image
import io

async def read_image_from_upload(file):
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")
