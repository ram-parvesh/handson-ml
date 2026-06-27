import argparse
import joblib
import pandas as pd
from config import MODEL_PATH, SCALER_PATH

MODEL_PATH = "models/diabetes_model.pkl"
SCALER_PATH = "models/diabetes_scaler.pkl"

parser = argparse.ArgumentParser(description="Diabetes Prediction")
parser.add_argument("--pregnancies", type=int, required=True )
parser.add_argument("--glucose", type=int, required=True)
parser.add_argument("--blood_pressure", type=int, required=True)
parser.add_argument("--skin_thickness", type=int, required=True)
parser.add_argument("--insulin", type=int, required=True)
parser.add_argument("--bmi", type=float, required=True)
parser.add_argument("--dpf", type=float, required=True)
parser.add_argument("--age", type=int, required=True)
args = parser.parse_args()

#load the saved model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

#important feature name must match

input_data = pd.DataFrame({
    "Pregnancies": [args.pregnancies],
    "Glucose": [args.glucose],
    "BloodPressure": [args.blood_pressure],
    "SkinThickness": [args.skin_thickness],
    "Insulin": [args.insulin],
    "BMI": [args.bmi],  
    "DiabetesPedigreeFunction": [args.dpf],
    "Age": [args.age]
})  

#scale and predict the target variable
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)   

if prediction[0] == 1:
    print("The patient is likely to have diabetes.")
else:
    print("The patient is unlikely to have diabetes.")  