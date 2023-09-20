import sys
import pandas as pd
from src.logger import logging as lg
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            lg.info('Initiating prediction')
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            scaled_data = preprocessor.transform(features)

            prediction = model.predict(scaled_data)
            lg.info('Prediction completed')

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,mean_radius,mean_perimeter,mean_area,mean_concavity,mean_concave_points,
                 worst_radius,worst_perimeter,worst_area,worst_concavity,worst_concave_points):
        self.mean_radius = mean_radius
        self.mean_perimeter = mean_perimeter
        self.mean_area = mean_area
        self.mean_concavity = mean_concavity
        self.mean_concave_points = mean_concave_points
        self.worst_radius = worst_radius
        self.worst_perimeter = worst_perimeter
        self.worst_area = worst_area
        self.worst_concavity = worst_concavity
        self.worst_concave_points = worst_concave_points
        
    def to_dataframe(self):
        try:
            lg.info('Creating dataframe')
            input_dict = {
                'mean radius':[self.mean_radius],
                'mean perimeter':[self.mean_perimeter],
                'mean area':[self.mean_area],
                'mean concavity':[self.mean_concavity],
                'mean concave points':[self.mean_concave_points],
                'worst radius':[self.worst_radius],
                'worst perimeter':[self.worst_perimeter],
                'worst area':[self.worst_area],
                'worst concavity':[self.worst_concavity],
                'worst concave points':[self.worst_concave_points]                
            }
            lg.info('Dataframe created')
            dataframe = pd.DataFrame(input_dict)
            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)
