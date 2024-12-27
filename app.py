# LungSense.AI
# Author: Adrian Simon
# API ENDPOINT FOR LUNG CANCER PREDICTION

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Loading of model
model = joblib.load('lungSenseModel.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to the questionnaire page
@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

# Route to process the questionnaire and display results from the model
@app.route('/submit', methods=['POST'])
def submit():
    data = [
        int(request.form['gender']),
        int(request.form['age']),
        int(request.form['smoking']),
        int(request.form['exercise']),
        int(request.form['family_history']),
        int(request.form['healthy_diet']),
        int(request.form['chronic_disease']),
        int(request.form['fatigue']),
        int(request.form['secondhand']),
        int(request.form['wheezing']),
        int(request.form['air_pollution']),
        int(request.form['coughing']),
        int(request.form['shortness_of_breath']),
        int(request.form['swallowing_difficulty']),
        int(request.form['chest_pain']),
    ]
    scaledData = scaler.transform([data])
    prediction = model.predict(scaledData)[0]
    confidence = model.predict_proba(scaledData)[0][prediction] * 100

    if prediction == 1:
        message = "You ARE at risk of lung cancer"
        riskLevel = "present"
    else:
        message = "You are NOT at risk of lung cancer"
        riskLevel = "absent"
    return render_template('results.html', message=message, confidence=round(confidence, 3), riskLevel=riskLevel)

# Mapping a specific url (/predict) to the predict function
@app.route('/predict', methods=['POST'])

# Function to predict risk of lung cancer and provide confidence score
def predict():
    # Fetches data and prepares it for prediction
    input_data = request.get_json()
    features = np.array(input_data['features']).reshape(1, -1)
    
    # Scaling of features to normalize data for model prediction
    scaled_features = scaler.transform(features)

    # Prediction and confidence score
    prediction = int(model.predict(scaled_features)[0])
    probability = model.predict_proba(scaled_features)[0]
    confidence = float(probability[prediction])

    # Returns prediction and confidence score in JSON format
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
