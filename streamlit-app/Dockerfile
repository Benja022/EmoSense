# Usa una imagen base oficial de Python
FROM python:3.12-slim

# Instala las dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia el archivo requirements.txt en el directorio de trabajo
COPY src/requirements.txt .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de la aplicación en el directorio de trabajo
COPY . .

# Expone el puerto en el que se ejecutará la aplicación Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación Streamlit
CMD ["streamlit", "run", "src/app2.py", "--server.port=8501", "--server.address=0.0.0.0"]