# Databricks notebook source
from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model01.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Experiance1 = int(request.form['Experiance'])
    Test_score1 = int(request.form['Test_score'])
    Interview_score1 = int(request.form['Interview_score'])

    arr = np.array([[Experiance1,Test_score1,Interview_score1]])
    pred = model.predict(arr)

    return render_template('index.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
