# Imagen base de Python
FROM python:3.10-slim

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt e instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copiar el resto del código fuente
COPY . .

# Exponer el puerto por defecto de Uvicorn
EXPOSE 8000

# Comando para iniciar la app con Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
