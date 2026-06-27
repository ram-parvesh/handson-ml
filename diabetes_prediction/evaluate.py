import joblib   
from config import MODEL_PATH, SCALER_PATH
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import load_and_preprocess_data

#load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

#load the saved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)   

#predict the target variable on the testing set
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)  

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}") 