import streamlit as st
import joblib
import librosa
import numpy as np
import soundfile as sf
import speech_recognition as sr
import datetime
import os
import base64

# Cargar los modelos y el vectorizador
svm_model = joblib.load('./streamlit-app/models/svm_model.pkl')
vectorizer = joblib.load('./streamlit-app/models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('./streamlit-app/models/label_encoder.pkl')
model_audio_mlp = joblib.load('./streamlit-app/models/model_emotions_audio_mlp.pkl')
label_encoder_audio_mlp = joblib.load('./streamlit-app/models/label_encoder_audio_mlp.pkl')

# Función para predecir la emoción de un texto
def predict_emotion_text(text):
    vectorized_text = vectorizer.transform([text])
    prediction = svm_model.predict(vectorized_text)
    emotion = label_encoder.inverse_transform(prediction)
    return emotion[0]

# Función para extraer características de audio
def extract_feature(file_name, max_pad_len=180):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if len(X) < sample_rate:
            X = np.pad(X, (0, max(0, sample_rate - len(X))), "constant")
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs.flatten()[:max_pad_len]

# Crear carpeta para guardar audios
output_folder = "Records"
os.makedirs(output_folder, exist_ok=True)

# Configuración de la imagen de fondo
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    page_bg_img = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("./streamlit-app/src/background4.png")

# Mostrar el logo en la parte superior
def show_logo(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    logo_html = f'''
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{encoded_string}" width="400">
    </div>
    '''
    st.markdown(logo_html, unsafe_allow_html=True)

show_logo("./streamlit-app/src/logo.png")

# Título de la aplicación
st.markdown("<h1 style='text-align: center;'>Predicción de Emociones</h1>", unsafe_allow_html=True)

# Entrada de texto
text_input = st.text_input("Introduce un texto:")

if st.button("Predecir Emoción del Texto"):
    if text_input:
        emocion_predicha_texto = predict_emotion_text(text_input)
        st.write(f"La emoción predicha para la frase '{text_input}' es: {emocion_predicha_texto}")
    else:
        st.write("Por favor, introduce una frase.")

# Captura de audio con SpeechRecognition
st.write("O usa tu micrófono para capturar una frase:")
if st.button("Capturar Audio"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Hablando... (Di 'salir' para finalizar)")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio, language="es-ES")
            st.write(f"📝 Texto detectado: {text}")

            # Guardar el audio en un archivo WAV
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_folder, f"grabacion_{timestamp}.wav")
            with open(filename, "wb") as f:
                f.write(audio.get_wav_data())
            st.write(f"✅ Audio guardado en: {filename}")

            # Predecir la emoción usando el modelo de texto
            emocion_predicha_texto = predict_emotion_text(text)
            st.write(f"🎭 Emoción predicha texto: {emocion_predicha_texto}")

            # Predecir la emoción usando el modelo de audio
            features = extract_feature(filename)
            if features is not None:
                features = features.reshape(1, -1)

                if model_audio_mlp is not None:
                    numeric_prediction_mlp = model_audio_mlp.predict(features)
                    predicted_emotion_audio_mlp = label_encoder_audio_mlp.inverse_transform(numeric_prediction_mlp)[0]
                    st.write(f"🎭 Emoción predicha audio: {predicted_emotion_audio_mlp}")
                else:
                    st.write("❌ Modelo MLP de audio no disponible.")

            else:
                st.write("❌ No se pudieron extraer características del audio.")

            # Determinar el resultado final
            if emocion_predicha_texto == "happy" and predicted_emotion_audio_mlp == "angry":
                resultado_final = "sarcasm"
            elif emocion_predicha_texto == "neutral":
                resultado_final = predicted_emotion_audio_mlp
            else:
                resultado_final = emocion_predicha_texto

            st.write(f"🔮 Resultado final: {resultado_final}")

        except sr.UnknownValueError:
            st.write("⚠️ No se entendió el audio, intenta de nuevo...")
        except sr.RequestError as e:
            st.write(f"❌ Error en la solicitud: {e}")
