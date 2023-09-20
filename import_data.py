import sys
import pandas as pd
from src.exception import CustomException
from src.utils import write_mongo

try:
    data = pd.read_csv("notebook/data/raw_breast_cancer.csv")
    write_mongo(data)    

except Exception as e:
    raise CustomException(e, sys)