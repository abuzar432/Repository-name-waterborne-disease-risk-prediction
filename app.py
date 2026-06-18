from flask import Flask, render_template, request
from datetime import datetime
import pickle
import numpy as np

import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "waterborne_model.pkl"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template(
    'index.html',
    temperature=0,
    do=0,
    ph=0,
    bod=0,
    rainfall=0
)


@app.route('/predict', methods=['POST'])
def predict():
    if (
    request.form['type_water_body'] == "" or
    request.form['state_name'] == ""
    ):
     return render_template(
        "index.html",
        error_message="Please select Water Body and State before prediction."
    )

    features = [
    float(request.form['type_water_body']),
    float(request.form['state_name']),
    float(request.form['min_temperature']),
    float(request.form['min_dissolved_oxygen']),
    float(request.form['min_ph']),

    100,     # conductivity
    float(request.form['min_bod']),
    0.5,     # nitrate
    50,     # fecal coliform
    400,     # total coliform
    2022,    # year
    90,      # toilet access
    10,      # open defecation
    float(request.form['rainfall'])
]

    final_features = np.array(features).reshape(1, -1)

    print("Features Sent:", final_features)

    prediction = model.predict(final_features)

    print("Prediction:", prediction)

    predicted_label = prediction[0]

    if predicted_label == 0:
        risk = "HIGH RISK"
        diseases = "Cholera, Typhoid Fever, Hepatitis A/E"

    elif predicted_label == 1:
        risk = "LOW RISK"
        diseases = "Mild Gastroenteritis"

    else:
        risk = "MEDIUM RISK"
        diseases = "Acute Diarrhoeal Disease, Dysentery"

    temperature = request.form['min_temperature']
    do = request.form['min_dissolved_oxygen']
    ph = request.form['min_ph']
    bod = request.form['min_bod']
    rainfall = request.form['rainfall']

    prediction_text = f"Predicted Risk Level : {risk}"
    disease_text = diseases

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M %p")

    return render_template(
    'index.html',
    prediction_text=prediction_text,
    disease_text=disease_text,

    type_water_body=request.form['type_water_body'],
    state_name=request.form['state_name'],

    temperature=temperature,
    do=do,
    ph=ph,
    bod=bod,
    rainfall=rainfall,

    current_time=current_time
)

if __name__ == "__main__":
    app.run(debug=True)