import logging
from data_preprocessing import load_and_preprocess_data
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from config import MODEL_DIR, MODEL_PATH, SCALER_PATH  

#logging configuration
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the training process...")

X_train, X_test, y_train, y_test = load_and_preprocess_data()

#Training the models
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
    "Support Vector Machine": SVC(kernel='linear', C=1.0, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}  

for name, model in classifiers.items():
    logging.info(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"{name} Accuracy: {accuracy}")
    logging.info(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    logging.info(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")       

#save artifacts
MODEL_DIR.mkdir(parents=True, exist_ok=True)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, SCALER_PATH)
best_model = classifiers["Logistic Regression"]  # Assuming Logistic Regression is the best model
joblib.dump(best_model, MODEL_PATH)

