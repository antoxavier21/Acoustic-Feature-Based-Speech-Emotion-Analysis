import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from main import load_data

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(x_train, y_train)
svm_y_pred = svm_model.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_y_pred)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(x_train, y_train_encoded)
xgb_y_pred = xgb_model.predict(x_test)
xgb_y_pred_decoded = label_encoder.inverse_transform(xgb_y_pred)
xgb_accuracy = accuracy_score(y_test, xgb_y_pred_decoded)

if __name__ == "__main__":
    print("Random Forest Accuracy: {:.2f}%".format(rf_accuracy * 100))
    print("\nClassification Report (Random Forest):")
    print(classification_report(y_test, rf_y_pred))
    print("\nConfusion Matrix (Random Forest):")
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    print(rf_cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        rf_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
    )
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    

    print("SVM Accuracy: {:.2f}%".format(svm_accuracy * 100))
    print("\nClassification Report (SVM):")
    print(classification_report(y_test, svm_y_pred))
    print("\nConfusion Matrix (SVM):")
    svm_cm = confusion_matrix(y_test, svm_y_pred)
    print(svm_cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        svm_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
    )
    plt.title('Confusion Matrix - SVM')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    
    print("XGBoost Accuracy: {:.2f}%".format(xgb_accuracy * 100))
    print("\nClassification Report (XGBoost):")
    print(classification_report(y_test, xgb_y_pred_decoded))
    print("\nConfusion Matrix (XGBoost):")
    xgb_cm = confusion_matrix(y_test, xgb_y_pred_decoded)
    print(xgb_cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        xgb_cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
    )
    plt.title('Confusion Matrix - XGBoost')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

