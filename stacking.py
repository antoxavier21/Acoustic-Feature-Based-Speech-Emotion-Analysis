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
from sklearn.linear_model import LogisticRegression
from different_models import svm_accuracy,rf_accuracy,xgb_accuracy
from main import load_data,accuracy

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

estimators = [
        ('mlp', MLPClassifier(
            alpha=0.01, batch_size=256, epsilon=1e-08,
            hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500
        )),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('svm', SVC(kernel='linear', random_state=42)),
        ('xgb', XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
    ]

stacking_model = make_pipeline(
        StandardScaler(),
        StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000))
    )

stacking_model.fit(x_train, y_train)

stacking_y_pred = stacking_model.predict(x_test)

stacking_accuracy = accuracy_score(y_test, stacking_y_pred)

if __name__ == "__main__":
    
    print("Stacking Classifier Accuracy: {:.2f}%".format(stacking_accuracy * 100))

    print("\nClassification Report (Stacking Classifier):")
    print(classification_report(y_test, stacking_y_pred))
    print("\nConfusion Matrix (Stacking Classifier):")
    cm = confusion_matrix(y_test, stacking_y_pred)
    print(cm)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test)
    )
    plt.title('Confusion Matrix - Stacking Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    
    accuracies = [accuracy, rf_accuracy, svm_accuracy, xgb_accuracy, stacking_accuracy]
    models = ['MLP Classifier', 'Random Forest', 'SVM', 'XGBoost', 'Stacking Classifier']

    plt.figure(figsize=(10, 6))
    sns.barplot(x=models, y=accuracies, palette='viridis')
    plt.title('Model Comparison (Accuracy)')
    plt.ylabel('Accuracy')
    plt.show()
