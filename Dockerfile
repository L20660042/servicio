# Usa la imagen oficial de Python desde Docker Hub
FROM python:3.9-slim

# Configura el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo de requerimientos y instala las dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia los archivos del proyecto al contenedor
COPY . .

# Expón el puerto en el que corre la app
EXPOSE 8000

# Ejecuta la aplicación con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
