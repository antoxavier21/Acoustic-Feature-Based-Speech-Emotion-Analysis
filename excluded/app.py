from sklearn.preprocessing import LabelEncoder
import streamlit as st
import soundfile as sf
import numpy as np
import os
import sounddevice as sd
import wavio
import time
from stacking import stacking_model, stacking_accuracy
from stacking1 import stacking_model as stack1, stacking_accuracy as acc
from tensorflow.keras.models import load_model # type: ignore
from main import emotions, extract_feature
from cnn1 import x_test,y_test
import joblib

UPLOAD_DIR = "audio_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

UPLOADED_FILE_PATH = os.path.join(UPLOAD_DIR, "uploaded_audio.wav")
RECORDED_FILE_PATH = os.path.join(UPLOAD_DIR, "recorded_audio.wav")

emotion_emoji_map = {
    'neutral': "😐",
    'calm': "😌",
    'happy': "😃",
    'sad': "😢",
    'angry': "😠",
    'fearful': "😨",
    'disgust': "🤢",
    'surprised': "😲"
}

st.title("🎭 Speech Emotion Recognition")
st.write("Upload an audio file or record your voice to predict the emotion!")

model_choice = st.selectbox("Choose Prediction Model", ["Stacking", "CNN", "Stacking and CNN"])

cnn_model = load_model("cnn_model_1.keras")
#cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Expected Input Shape:", cnn_model.input_shape)

label_encoder = LabelEncoder()
label_encoder.fit(['calm', 'disgust', 'fearful', 'happy'])

def predict_emotion_stacking(audio_path):
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
    prediction = stacking_model.predict([features])
    return prediction[0]

ml_dl_model = joblib.load("ml_and_dl.pkl")

def predict_emotion_stacking1(audio_path):
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)

    print("\nExtracted Features (First 10 values):", features[:10])
    print("Feature Length:", len(features))

    expected_length = 128  
    if len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)), mode='constant')  
    elif len(features) > expected_length:
        features = features[:expected_length]  

    features = np.array(features).reshape(1, -1)  

    print("Final Feature Shape for Stacking Classifier Prediction:", features.shape)

    predicted_label_index = ml_dl_model.predict(features)[0]  

    print("Stacking Model Predicted Label Index:", predicted_label_index)

    predicted_emotion = label_encoder.inverse_transform([predicted_label_index])[0]  

    print(f"Predicted Emotion (Stacking Model): {predicted_emotion}")

    return predicted_emotion



def predict_emotion_cnn(audio_path):
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)

    print("\nExtracted Features (First 10 values):", features[:10]) 
    print("Feature Length:", len(features))

    expected_length = 128  
    if len(features) < expected_length:
        features = np.pad(features, (0, expected_length - len(features)), mode='constant')  
    elif len(features) > expected_length:
        features = features[:expected_length] 

    features = np.expand_dims(features, axis=0)  
    features = np.expand_dims(features, axis=-1)  

    print("Final Feature Shape for Prediction:", features.shape)

    prediction = cnn_model.predict(features)

    print("\nRaw Model Prediction Probabilities:", prediction)

    predicted_label = np.argmax(prediction, axis=1)[0]
    print("CNN Predicted Label Index:", predicted_label)

    emotion_label = label_encoder.classes_[predicted_label]
    print(f"Predicted Emotion: {emotion_label}")

    return emotion_label


_,cnn_accuracy = cnn_model.evaluate(x_test, y_test, verbose=2)
uploaded_file = st.file_uploader("📂 Upload an audio file", type=["wav"])

if uploaded_file is not None:
    with open(UPLOADED_FILE_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())  

    if model_choice == "Stacking":
        emotion_name = predict_emotion_stacking(UPLOADED_FILE_PATH)
        model_accuracy = stacking_accuracy * 100
    elif model_choice == "CNN":
        emotion_name = predict_emotion_cnn(UPLOADED_FILE_PATH)
        model_accuracy = cnn_accuracy * 100
    else:
        emotion_name = predict_emotion_stacking1(UPLOADED_FILE_PATH)
        model_accuracy = acc * 100
   
    st.write(f"**Predicted Emotion (Uploaded Audio):** {emotion_name} {emotion_emoji_map.get(emotion_name, '❓')}")
    st.write(f"**Model Accuracy:** {model_accuracy if isinstance(model_accuracy, str) else f'{model_accuracy:.2f}%'}")


def record_audio(filename, duration=5, fs=44100):
    """Record audio from the microphone and save it."""
    st.write("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    wavio.write(filename, recording, fs, sampwidth=2)  
    st.write("✅ Recording complete!")

if st.button("Record Live Audio"):
    record_audio(RECORDED_FILE_PATH)

    if model_choice == "Stacking":
        emotion_name = predict_emotion_stacking(RECORDED_FILE_PATH)
        model_accuracy = stacking_accuracy * 100
    elif model_choice == "CNN":
        emotion_name = predict_emotion_cnn(RECORDED_FILE_PATH)
        model_accuracy = cnn_accuracy * 100
    else:
        emotion_name = predict_emotion_stacking1(UPLOADED_FILE_PATH)
        model_accuracy = acc * 100

    st.write(f"**Predicted Emotion (Recorded Audio):** {emotion_name} {emotion_emoji_map.get(emotion_name, '❓')}")
    st.write(f"**Model Accuracy:** {model_accuracy if isinstance(model_accuracy, str) else f'{model_accuracy:.2f}%'}")

