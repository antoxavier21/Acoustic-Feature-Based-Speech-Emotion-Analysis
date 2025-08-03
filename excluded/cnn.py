import keras
import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ReduceLROnPlateau # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# Function to extract mel spectrogram features for CNN
def extract_feature_cnn(file_name,mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

def load_data_cnn(test_size=0.2, max_len=128):
    x, y = [], []
    for file in glob.glob("C:\\Users\\Anto\\Downloads\\speech-emotion-recognition-ravdess-data\\ravdess data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2].zfill(2), "unknown")
        if emotion not in observed_emotions:
            continue
        feature = extract_feature_cnn(file, mfcc=True, chroma=True, mel=True)
        
        if len(feature) < max_len:
            feature = np.pad(feature, (0, max_len - len(feature)), mode='constant')
        elif len(feature) > max_len:
            feature = feature[:max_len]
        
        x.append(feature)
        y.append(emotion)
    
    x = np.array(x)
    y = np.array(y)
    return train_test_split(x, y, test_size=test_size, random_state=9)

x_train, x_test, y_train, y_test = load_data_cnn(test_size=0.25)

x_train = x_train[..., np.newaxis] 
x_test = x_test[..., np.newaxis]

label_encoder = LabelEncoder()
y_train = to_categorical(label_encoder.fit_transform(y_train))
y_test = to_categorical(label_encoder.transform(y_test))


cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(128, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='softmax')
])


cnn_model.compile(
    optimizer='adam',  
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train.argmax(axis=1)),
    y=y_train.argmax(axis=1)
)

class_weight_dict = dict(enumerate(class_weights))
print("Class Weights:", class_weight_dict)


if __name__ == "__main__":
 
    history = cnn_model.fit(x_train, y_train, epochs=70, batch_size=40, validation_data=(x_test, y_test),class_weight=class_weight_dict)

    
    test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.show()

    y_pred = cnn_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    from sklearn.metrics import confusion_matrix, classification_report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    '''cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    keras.saving.save_model(cnn_model,"cnn_model.keras")
    print("Model saved successfully!")
    '''
