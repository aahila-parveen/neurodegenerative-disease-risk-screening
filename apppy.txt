from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

df = pd.read_csv("dataset/neuro_synthetic_dataset.csv")

print(df["Disease"].value_counts())

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    memory = int(request.form["memory"])
    movement = int(request.form["movement"])
    speech = int(request.form["speech"])
    family = int(request.form["family"])
    duration = int(request.form["duration"])

    features = np.array([[age, memory, movement, speech, family, duration]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    risk_score = round(probability * 100, 2)

    diseases = {
    0: "Alzheimer-like",
    1: "Huntington-like",
    2: "Parkinson-like"
}

    primary_risk = diseases.get(prediction, "Low Neurological Risk Pattern")

    # Risk level
    if risk_score < 40:
        risk_level = "Low"
    elif risk_score < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"
