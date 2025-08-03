from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from cnn1 import cnn_model, label_encoder, x_train, x_test, y_train, y_test  

class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(len(label_encoder.classes_))  

    def fit(self, X, y):
        self.classes_ = np.unique(y)  
        return self  

    def predict(self, X):
        X = np.expand_dims(X, axis=-1)  
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)  

    def predict_proba(self, X):
        X = np.expand_dims(X, axis=-1)
        return self.model.predict(X, verbose=0)  


cnn_wrapper = CNNWrapper(cnn_model)

estimators = [
    ('mlp', MLPClassifier(
        alpha=0.01, batch_size=256, epsilon=1e-08,
        hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500
    )),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)),
    ('cnn', cnn_wrapper)  
]

x_train_reshaped = x_train.reshape(x_train.shape[0], -1)
x_test_reshaped = x_test.reshape(x_test.shape[0], -1)

stacking_model = make_pipeline(
    StandardScaler(),
    StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

stacking_model.fit(x_train_reshaped, y_train_labels)

stacking_y_pred = stacking_model.predict(x_test_reshaped)


stacking_model.fit(x_train_reshaped, y_train_labels)

stacking_y_pred = stacking_model.predict(x_test_reshaped)

stacking_accuracy = accuracy_score(y_test_labels, stacking_y_pred)

if __name__ == "__main__":
    print("Stacking Classifier Accuracy: {:.2f}%".format(stacking_accuracy * 100))

    print("\nClassification Report (Stacking Classifier):")
    print(classification_report(y_test_labels, stacking_y_pred))

    print("\nConfusion Matrix (Stacking Classifier):")
    cm = confusion_matrix(y_test_labels, stacking_y_pred)
    print(cm)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix - Stacking Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

#import joblib

#scaler = StandardScaler()
#scaler.fit(x_train_reshaped)

#joblib.dump(stacking_model, "ml_and_dl.pkl")
#joblib.dump(scaler, "scaler.pkl")