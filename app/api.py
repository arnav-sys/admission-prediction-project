# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:26:30 2022

@author: Surface
"""


from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

model = joblib.load("../model.sav")

app = Flask(__name__)


@app.route('/', methods = ['GET'])
def MainPage():
    return "<h1>App</h1>"

@app.route("/model", methods = ["POST"])
def run_model():
    if request.method == "POST":
        data = request.get_json()
        df= pd.json_normalize(data)
        Std = StandardScaler()
        df2 =pd.read_csv("../Admission_Prediction.csv")
        Std.fit(df2.drop(columns=["Serial No.","Chance of Admit"]))
        df = Std.transform(df)
        predictions = model.predict(df)
        
        return str(predictions)