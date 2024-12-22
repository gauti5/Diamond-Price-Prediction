import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path:str=os.path.join('Artifacts',"raw_data.csv")
    train_data_path:str=os.path.join('Artifacts',"train_data.csv")
    test_data_path:str=os.path.join('Artifacts',"test_data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started!")
        
        try:
            data = pd.read_csv('Notebook/Data/gemstone.csv')
            logging.info("Read the data from CSV file")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data Created!")
            
            logging.info("Splitting the data into train and test")
            
            train_data, test_data=train_test_split(data, test_size=0.3, random_state=23)
            logging.info("Data Splitting is Done!")
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("train data and test date created!")
            logging.info("Data Ingestion Completed!!")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exception occured during data ingestion")
            
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    train_arr, test_arr=data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))
    
    
    


