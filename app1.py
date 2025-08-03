from sklearn.preprocessing import LabelEncoder
import streamlit as st
import numpy as np
import os
import sounddevice as sd
import wavio
from stacking import stacking_model, stacking_accuracy
from stacking1 import stacking_model as stack1, stacking_accuracy as acc
from tensorflow.keras.models import load_model # type: ignore
from main import extract_feature,observed_emotions
from cnn1 import x_test,y_test

UPLOAD_DIR = "audio_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

UPLOADED_FILE_PATH = os.path.join(UPLOAD_DIR, "uploaded_audio.wav")
RECORDED_FILE_PATH = os.path.join(UPLOAD_DIR, "recorded_audio.wav")


emotion_image_map = {
    'neutral': "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSpWooUBeHJa7f2LYkgpP0EqnacUY-yfUKicg&s",  
    'calm': "https://images.emojiterra.com/google/noto-emoji/animated-emoji/1f60c.gif",      
    'happy': "https://media4.giphy.com/media/QWvra259h4LCvdJnxP/giphy.gif?cid=6c09b952sbxyuey4xcitwtowcw5e900efedv9yuebol2tu07&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=e",     
    'sad': "https://media1.giphy.com/media/h4OGa0npayrJX2NRPT/giphy.gif?cid=6c09b95204y98325psfi3gip8jazxbawa6g6gwwmrtc1j1lz&ep=v1_internal_gif_by_id&rid=giphy.gif&ct=e",       
    'angry': "https://i.pinimg.com/originals/c4/07/04/c4070448ce8174b2b3e121081ddbbee5.gif",     
    'fearful': "https://media0.giphy.com/media/XEyXIfu7IRQivZl1Mw/giphy.gif?cid=6c09b952g338evtko95bebt7wox3sh00x2gocv16zmbot8rh&ep=v1_stickers_search&rid=giphy.gif&ct=e",   
    'disgust': "https://i.pinimg.com/originals/75/b3/a3/75b3a3b3d4b3bc4d1f9e1ae357c2e5ed.gif",   
    'surprised': "https://i.pinimg.com/originals/7a/09/28/7a092873bc2103165ba7d17ab031281f.gif"  
}

st.markdown(
    """
    <style>
        .main {background-color: #0d0202;}
        .title {color: #ffffff; text-align: center; padding: 10px; border-radius: 8px; background: linear-gradient(to right, #4A90E2, #9013FE);}
        .upload-box {text-align: center; background-color: #fff; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 15px rgba(0,0,0,0.1);}
        .prediction-card {background: black; padding: 15px; border-radius: 10px; box-shadow: 3px 3px 10px rgba(0,0,0,0.1); text-align: center;}
        .record-btn {background-color: #ff4b4b; color: white; border-radius: 5px; padding: 8px 16px; font-size: 16px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">🎭 Speech Emotion Recognition</h1>', unsafe_allow_html=True)

st.write("Upload an audio file or record your voice to predict the emotion!")

model_choice = st.selectbox("Choose Prediction Model", ["Stacking", "CNN", "Stacking and CNN"])

cnn_model = load_model("cnn_model_1.keras")
#cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Expected Input Shape:", cnn_model.input_shape)
import joblib

label_encoder = LabelEncoder()
label_encoder.fit(observed_emotions)

def predict_emotion_stacking(audio_path):
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
    prediction = stacking_model.predict([features])
    return prediction[0]

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

    predicted_label_index = stack1.predict(features)[0]  

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
   
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Predicted Emotion: {emotion_name} </h3>
        <img src="{emotion_image_map.get(emotion_name, 'https://i.imgur.com/q2zBnAS.png')}" width="120">
        <p><b>Model Accuracy:</b> {model_accuracy:.2f}%</p>
    </div>
""", unsafe_allow_html=True)
    st.write(f"**Model Accuracy:** {model_accuracy if isinstance(model_accuracy, str) else f'{model_accuracy:.2f}%'}")


import soundfile as sf

def record_audio(filename, duration=5, fs=16000):  
    st.write("🎙️ Recording... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, fs)  
    st.write("✅ Recording complete!")

if st.button("🎤 Record Live Audio", key="record_audio", help="Click to record"):
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

    st.markdown(f"""
    <div class="prediction-card">
        <h3>Predicted Emotion: {emotion_name} </h3>
        <img src="{emotion_image_map.get(emotion_name, 'https://i.imgur.com/q2zBnAS.png')}" width="120">
        <p><b>Model Accuracy:</b> {model_accuracy:.2f}%</p>
    </div>
""", unsafe_allow_html=True)
    st.write(f"**Model Accuracy:** {model_accuracy if isinstance(model_accuracy, str) else f'{model_accuracy:.2f}%'}")

