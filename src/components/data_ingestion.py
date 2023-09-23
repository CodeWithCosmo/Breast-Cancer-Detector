import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging as lg
from src.utils import read_mongo

@dataclass
class DataIngestionConfig:
    train_dataset_path: str = os.path.join('artifacts','train.csv')
    test_dataset_path: str = os.path.join('artifacts','test.csv')
    raw_dataset_path: str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        try:
            self.ingestion_config = DataIngestionConfig()
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self):
        lg.info('Initiating data ingestion')
        try:
            lg.info('Downloading data from MongoDB Cloud')
            data = pd.DataFrame(read_mongo())
            lg.info('Dowloading successful')
            # data = pd.read_csv('notebook/data/StudentsPerformance.csv') #? Local Source
            #~ This is the method where we can change the data ingestion source
            lg.info('Data ingestion completed')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_dataset_path), exist_ok=True)
            lg.info('Feature Selection initiated')
            corr_matrix = data.corr()
            threshold = 0.6
            selected_columns = []
            # Iterate through the correlation matrix
            for column in data.columns:
                # Exclude the column itself (correlation with itself is always 1.0)
                if column != 'target': 
                    # Check if the absolute correlation is greater than the threshold
                    if abs(corr_matrix['target'][column]) > threshold:
                        selected_columns.append(column)
            # Extract the selected columns from the DataFrame
            data = pd.concat([data[selected_columns], data['target']], axis=1)
            lg.info('Feature Selection completed')

            data.to_csv(self.ingestion_config.raw_dataset_path, index=False, header=True)
            lg.info('Train test split initiated')
            train_set,test_set = train_test_split(data,test_size=0.25,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_dataset_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_dataset_path, index=False, header=True)

            lg.info('Train test split completed')

            return (
                self.ingestion_config.train_dataset_path,
                self.ingestion_config.test_dataset_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        