from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

model = pickle.load(open('LogisticRegressionHeart.sav', 'rb'))
cols = ['age', 'sex','cp',	'trestbps',	'chol',	'fbs',	'restecg',	'thalach',	'exang',	'oldpeak',	'slope',	'ca',	'thal']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    final=[final]
    prediction = model.predict(final)
    prediction = int(prediction)
    return render_template('home.html',pred='The report of your heart is  {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict(data)
    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
