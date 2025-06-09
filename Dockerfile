# Usar imagen base de Python
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requisitos y el código
COPY requirements.txt ./
COPY main.py ./


# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto
EXPOSE 8000

# Comando para correr el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
