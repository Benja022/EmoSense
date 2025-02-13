# 📢 EmoSense

## 1️⃣ Introducción
En una sociedad donde la inteligencia emocional, la salud mental y los estados emocionales son cada vez más importantes, las empresas necesitan adaptarse para mejorar la experiencia del usuario y la fidelización de clientes. 💡💖

EmoSense es una aplicación basada en Machine Learning 🤖 que detecta emociones a través de canales de comunicación, tanto verbal como escrito. Analiza:
- 🎵 **Ondas y espectrogramas del audio**.
- 📝 **Sentido semántico de las palabras**.

## 2️⃣ Planteamiento del Problema
Las empresas con gran cantidad de clientes (✈️ aerolíneas, 🚆 trenes, 🛒 ventas a gran escala) buscan **automatizar** la atención sin perder calidad. Muchos usuarios aún son reacios a interactuar con máquinas. 🤯

Añadiendo **análisis de emociones** a estos métodos automáticos, podemos **mejorar la experiencia** y **aumentar la fidelización**. 🎯😊

## 3️⃣ Estado Actual
En nuestra investigación encontramos dos enfoques:
1. 📜 **Análisis semántico**: Speech-to-text para convertir audio en texto y analizar las palabras.
2. 🎙️ **Análisis de audio**: Ondas, tono y espectrogramas para deducir emociones.

### 🧠 Algoritmos encontrados
- Modelos tradicionales: 🌲 Random Forest, 🌳 Decision Tree.
- Modelos avanzados: 🤖 Deep Neural Networks (DNNs), 🧩 Convolutional Neural Networks (CNNs).

Para nuestra aplicación probaremos varios modelos y seleccionaremos el más adecuado. ⚙️📊

## 4️⃣ Metodología
### 📂 Fuente de Datos
📌 **RAVDESS Emotional Speech Audio**
🔗 [Dataset en Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

Las emociones están etiquetadas en los nombres de los archivos:
- 😐 Neutral (01)
- 😌 Calmado (02)
- 😃 Feliz (03)
- 😢 Triste (04)
- 😡 Enojado (05)
- 😨 Temeroso (06)
- 🤢 Disgustado (07)
- 😲 Sorprendido (08)

También se incluyen otras características como **intensidad y actor**. 🎭

### 🔍 Preprocesamiento de Datos
- 📌 Limpieza y normalización.
- 📈 Aumento de datos (Data Augmentation).

### ⚙️ Modelo de Machine Learning
- Algoritmos utilizados: 🧠 CNN, RNN, Transformers.
- Arquitectura del modelo: 🔄 Capas, funciones de activación, etc.
- Herramientas y frameworks: 🛠️ TensorFlow, PyTorch, Scikit-Learn.

### 🎯 Entrenamiento y Validación
- División de datos: 📊 Train / Validation / Test.
- Métricas de evaluación: 🎯 Precisión, Recall, F1-score.
- Mejora del rendimiento: 🔄 Data Augmentation, Transfer Learning, Regularización.

## 5️⃣ Resultados 📊
- 📈 Desempeño del modelo en pruebas.
- ⚖️ Comparación con otros enfoques.
- 🔍 Ejemplos de predicciones correctas e incorrectas.

## 6️⃣ Conclusiones y Futuro Trabajo 🚀
- 🔎 **Hallazgos**: ¿Qué hemos aprendido?
- ⚠️ **Limitaciones**: ¿Qué dificultades encontramos?
- 🛠️ **Mejoras futuras**: Expansión del dataset, optimización del modelo, implementación en producción.

## 7️⃣ Referencias 📚
📖 **Implementing Machine Learning Techniques for Continuous Emotion Prediction from Uniformly Segmented Voice Recordings**
🔗 [Artículo en Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1300996/full)


