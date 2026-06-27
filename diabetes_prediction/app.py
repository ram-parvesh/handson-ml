import numpy as np

filename = 'models/diabetes_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json(force=True)
    pregnancies = data['Pregnancies']
    glucose = data['Glucose']
    blood_pressure = data['BloodPressure']
    skin_thickness = data['SkinThickness']
    insulin = data['Insulin']
    bmi = data['BMI']
    dpf = data['DiabetesPedigreeFunction']
    age = data['Age']

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = classifier.predict(input_data)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})  


if __name__ == '__main__':
    app.run(debug=True)
