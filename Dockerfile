# Usa una imagen base de Python
FROM python:3.9-slim

# Instalar Tesseract OCR
RUN apt-get update && apt-get install -y tesseract-ocr

# Instalar las dependencias
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Expone el puerto donde FastAPI correrá
EXPOSE 8000

# Comando para ejecutar FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
