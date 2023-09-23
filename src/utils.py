import os
import sys
import dill
from py_dotenv import dotenv
from pymongo import MongoClient

from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging as lg
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)          

            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) 
    
        
def read_mongo():
    try:
            lg.info('Connecting to MongoDB Cloud')
            dotenv.read_dotenv('.env')
            client = os.getenv('client')
            lg.info('Connection successful')
            
            database = os.getenv('database')
            collection = os.getenv('collection')            

            client = MongoClient(client)
            db = client[database]
            collection = db[collection]
            cursor = collection.find({}) 
            data = list(cursor)
            client.close()
            return data
    
    except Exception as e:
        raise CustomException(e,sys)
def write_mongo(data):
    try:
            lg.info('Connecting to MongoDB Cloud')
            dotenv.read_dotenv('.env')
            client = os.getenv('client')
            lg.info('Connection successful')
            
            database = os.getenv('database')
            collection = os.getenv('collection')            

            client = MongoClient(client)
            db = client[database]
            collection = db[collection]
            lg.info('Inserting data into MongoDB Cloud')            
            data['_id'] = range(1, len(data) + 1)
            data = data.to_dict(orient='records')
            collection.insert_many(data)
            lg.info('Data insertion successful')
            client.close()            
    
    except Exception as e:
        raise CustomException(e,sys)