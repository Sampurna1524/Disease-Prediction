from flask import Flask, render_template, request
import joblib
import pandas as pd
import re

model = joblib.load('disease_prediction_model.pkl')
with open('symptom_list.txt', 'r') as f:
    all_symptoms = [line.strip() for line in f.readlines()] 

symptom_keywords = {s.replace('_', ' '): s for s in all_symptoms}

app = Flask(__name__)

def preprocess_input(text):
    text = text.lower()
    detected_symptoms = set()

    for keyword, original_symptom in symptom_keywords.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text):
            detected_symptoms.add(original_symptom)

    
    binary_input = [1 if symptom in detected_symptoms else 0 for symptom in all_symptoms]
    return binary_input

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['symptom_sentence']
    features = preprocess_input(sentence)
    df = pd.DataFrame([features], columns=all_symptoms) 
    prediction = model.predict(df)[0]
    return render_template('index.html', prediction=f"Predicted Disease: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
