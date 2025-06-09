# Dockerfile
FROM python:3.9-slim

# Instalar dependencias necesarias para OpenCV (si es necesario)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Crear un directorio de trabajo en el contenedor
WORKDIR /app

# Copiar todos los archivos del proyecto al contenedor
COPY . /app/

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch (asegurarse de la versión adecuada, por ejemplo, para CPU)
RUN pip install torch==1.9.1  # O usa la versión que necesitas para tu entorno (CPU o GPU)

# Exponer el puerto donde FastAPI escuchará
EXPOSE 8000

# Comando para ejecutar el servicio con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
