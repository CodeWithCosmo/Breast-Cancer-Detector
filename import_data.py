import sys
import pandas as pd
from pymongo import MongoClient
from src.logger import logging as lg
from src.exception import CustomException

try:
    lg.info("Connecting to MongoDB Cloud")
    client = MongoClient("mongodb+srv://Kaggler:mongocloud@cluster0.d110adr.mongodb.net/")
    lg.info('Connection successful')
    db = client["CancerDB"]
    collection = db["BreastCancer"]
    lg.info('Inserting data into MongoDB Cloud')
    data = pd.read_csv("notebook/data/raw_breast_cancer.csv")
    data['_id'] = range(1, len(data) + 1)
    data = data.to_dict(orient='records')
    collection.insert_many(data)
    lg.info('Data insertion successful')

except Exception as e:
    raise CustomException(e, sys)

client.close()