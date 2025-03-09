import streamlit as st
import joblib
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import wavio

# Cargar los modelos y el vectorizador
svm_model = joblib.load('models/svm_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Función para predecir la emoción de un texto
def predict_emotion_text(text):
    vectorized_text = vectorizer.transform([text])
    prediction = svm_model.predict(vectorized_text)
    emotion = label_encoder.inverse_transform(prediction)
    return emotion[0]

# Función para extraer características de audio
def extract_feature(file_name, max_pad_len=174):
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
        return mfccs.flatten()

# Función para predecir la emoción de un audio
def predict_emotion_audio(file_path):
    features = extract_feature(file_path).reshape(1, -1)
    prediction = svm_model.predict(features)
    emotion = label_encoder.inverse_transform(prediction)
    return emotion[0]

# Función para grabar audio
def record_audio(duration, filename):
    fs = 44100  # Sample rate
    st.write("Grabando...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)
    st.write("Grabación completada")

# Configuración de la aplicación Streamlit
st.title("EmoSense: Predicción de Emociones")

# Selección de tipo de entrada
input_type = st.radio("Selecciona el tipo de entrada:", ("Texto", "Audio"))

if input_type == "Texto":
    text_input = st.text_area("Introduce el texto:")
    if st.button("Predecir Emoción"):
        if text_input:
            emotion = predict_emotion_text(text_input)
            st.write(f"La emoción predicha es: {emotion}")
        else:
            st.write("Por favor, introduce un texto.")

elif input_type == "Audio":
    audio_file = st.file_uploader("Sube un archivo de audio (.wav):", type=["wav"])
    if st.button("Predecir Emoción"):
        if audio_file:
            with open("temp.wav", "wb") as f:
                f.write(audio_file.getbuffer())
            emotion = predict_emotion_audio("temp.wav")
            st.write(f"La emoción predicha es: {emotion}")
        else:
            st.write("Por favor, sube un archivo de audio.")
    
    st.write("O graba un nuevo audio:")
    if st.button("Grabar Audio"):
        record_audio(5, "temp.wav")
        emotion = predict_emotion_audio("temp.wav")
        st.write(f"La emoción predicha es: {emotion}")