import pickle
import sys
from flask import Flask, request,render_template

# import numpy as np
# import pandas as pd


from src.logger import logging as lg
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictionPipeline,CustomData

# from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predict", methods=['POST'])
def predict():  
         try: 
            data = CustomData(
                mean_radius = request.form.get('mean radius'),
                mean_perimeter = request.form.get('mean perimeter'),
                mean_area = request.form.get('mean area'),
                mean_concavity = request.form.get('mean concavity'),
                mean_concavity_points = request.form.get('mean concavity points'),
                worst_radius = request.form.get('worst radius'),
                worst_perimeter = request.form.get('worst perimeter'),  
                worst_area = request.form.get('worst area'),
                worst_concavity = request.form.get('worst concavity'),
                worst_concavity_points = request.form.get('worst concavity points')      
            )
            pred_df = data.to_dataframe()
            #  print(pred_df)

            predict_pipeline = PredictionPipeline()
            result = predict_pipeline.predict(pred_df)
            if result == 0:
                return render_template("home.html",prediction_text=f"This patient has Benign Tumor which is not cancerous.")
            else:
                 return render_template("home.html",prediction_text=f"This patient has Malignant Tumor which is cancerous.") 
         
         except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    lg.info('Application started')
    app.run()