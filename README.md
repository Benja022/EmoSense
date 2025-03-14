
# 📢 EmoSense

## 1️⃣ Introduction  

In a society where emotional intelligence, mental health, and emotional states are increasingly important, companies need to adapt to improve user experience and customer loyalty. 💡💖  

EmoSense is a Machine Learning-based application 🤖 that detects emotions through communication channels, both verbal and written. It analyzes:  

- 🎵 **Audio waves and spectrograms**.  

- 📝 **The semantic meaning of words**.  

## 2️⃣ Problem Statement  

Companies with a large number of customers (✈️ airlines, 🚆 trains, 🛒 large-scale sales) seek to **automate** customer service without losing quality. Many users are still reluctant to interact with machines. 🤯  

By adding **emotion analysis** to these automated methods, we can **enhance the experience** and **increase customer loyalty**. 🎯😊  

## 3️⃣ Current State  

In our research, we found two approaches:  

1. 📜 **Semantic analysis**: Speech-to-text to convert audio into text and analyze the words.  

1. 🎙️ **Audio analysis**: Waves, tone, and spectrograms to deduce emotions.  

### 🧠 Algorithms Found  

- Traditional models: 🌲 Random Forest, 🌳 Decision Tree.  
- Advanced models: 🤖 Deep Neural Networks (DNNs), 🧩 Convolutional Neural Networks (CNNs).  

For our application, we will test various models and select the most suitable one. ⚙️📊  

## 4️⃣ Methodology  

### 📂 Data Source  

📌 **RAVDESS Emotional Speech Audio**  
🔗 [Dataset on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)  

The emotions are labeled in the filenames:  

- 😐 Neutral (01)  
- 😌 Calm (02)  
- 😃 Happy (03)  
- 😢 Sad (04)  
- 😡 Angry (05)  
- 😨 Fearful (06)  
- 🤢 Disgusted (07)  
- 😲 Surprised (08)  

Other characteristics such as **intensity and actor** are also included. 🎭  

### 🔍 Data Preprocessing  

- 📌 Cleaning and normalization.  
- 📈 Data augmentation.  

### ⚙️ Machine Learning Model  

- Algorithms used: 🧠 MLP, SVM, CNN.  
- Tools and frameworks: 🛠️ TensorFlow, Librosa, Scikit-Learn.  

### 🎯 Training and Validation  

- Data split: 📊 Train / Validation / Test.  
- Evaluation metrics: 🎯 Accuracy, Confusion Matrix.  
- Performance improvement: 🔄 Class rebalancing with SMOTE, Model optimization with GridSearch.  

## 5️⃣ Results 📊  

- A Streamlit app that predicts emotions after recording an audio or writing a sentence.  

## 6️⃣ Conclusions and Future Work 🚀  

- Text-based emotion recognition has proven to be highly effective.  
- Audio-based emotion recognition presents unique challenges (finding datasets in Spanish, long training time, data quality).  
- In the weighting of the emotion decision, the prediction with text has more influence.  

## 6️⃣ Future Improvements 🚀  

- Expand training data to Spanish audio.  
- Integrate English text-based emotion recognition.  
- Real-time feedback loop for model improvement.  
- Data augmentation to enhance the model.  
- More emotions.  
- The emotion decision will be weighted by the model's accuracy.  

## 7️⃣ References 📚  

📖 **Implementing Machine Learning Techniques for Continuous Emotion Prediction from Uniformly Segmented Voice Recordings**
🔗 [Artículo en Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1300996/full)

## 8️⃣ Run the streamlit app 💻

You have two options to run the app:

1. If you have streamlit installed. Run the app with the following steps:

    1.1. Go to `streamlit-app/src` folder.

    1.2. Run the following command:

    ```bash
    streamlit run app.py
    ```

2. Run the app with docker:

    2.1. Go to `streamlit-app` folder and run the following command:
    docker build -t streamlit-app .
    2.2. Run the docker container with the following command:

```bash
docker run -p 8501:8501 streamlit-app
```

Then you can access the app at [http://localhost:8501](http://localhost:8501).

In `streamlit-app/src` folder you can find the `app.py` file where the app is defined to run with streamlit in your local machine. And `app2.py` file where the app is defined to run with docker.
