import streamlit as st
import speech_recognition as sr
from streamlit_audiorecorder import st_audiorecorder
import numpy as np
import io
import soundfile as sf

st.write("🎤 Usa tu micrófono para capturar una frase:")

# Capturar audio desde el micrófono sin guardarlo en un archivo
audio_bytes = st_audiorecorder("Presiona para hablar", key="audio")

if audio_bytes is not None:
    st.write("🔊 Procesando el audio...")

    # Convertir los bytes de audio a un formato compatible con SpeechRecognition
    audio_data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
    recognizer = sr.Recognizer()

    # Convertir el audio a un objeto de SpeechRecognition
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="es-ES")
        st.write(f"📝 Texto detectado: {text}")

        # Aquí puedes usar el texto para predecir la emoción con tu modelo
        # emocion_predicha_texto = predict_emotion_text(text)
        # st.write(f"🎭 Emoción predicha: {emocion_predicha_texto}")

    except sr.UnknownValueError:
        st.write("⚠️ No se entendió el audio, intenta de nuevo...")
    except sr.RequestError as e:
        st.write(f"❌ Error en la solicitud: {e}")
