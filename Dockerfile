# Usa una imagen base de Python
FROM python:3.9-slim

# Instala las dependencias
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código de tu aplicación
COPY . .

# Expone el puerto donde FastAPI correrá
EXPOSE 8000

# Comando para ejecutar FastAPI con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
