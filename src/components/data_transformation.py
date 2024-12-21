import os
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('Artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformation()
        
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation Initiated!!")
            
            numerical_cols=['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
            cat_cols=['cut', 'color', 'clarity']
            
            cut_categories=['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
            color_categories=['G','E', 'F', 'H', 'D', 'I', 'J']
            clarity_categories=['SI1','VS2', 'VS1', 'SI2', 'VVS2', 'VVS1', 'IF', 'I1']
            
            logging.info('Pipeline Started!!')
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )
            
            preproccesor=ColumnTransformer([
                ('num pipeline', num_pipeline, numerical_cols),
                ('cat pipeline', cat_pipeline, cat_cols)
            ])
            return preproccesor
        
        except Exception as e:
            logging.info("Exception occured while data transformation")
            
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transformation()
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            target_column_name='price'
            drop_columns=[target_column_name,'id']
            
            input_features_train_df=train_df.drop(columns=drop_columns, axis=1)
            target_features_train_df=train_df[target_column_name]
            
            input_features_test_df=test_df.drop(columns=drop_columns)
            target_features_test_df=test_df[target_column_name]
            
            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)
            
            logging.info('Applying the Preprocessor Object for train and test data')
            
            train_arr=np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr=np.c_[input_features_test_arr, np.array(target_features_test_df)]
            
            return train_arr, test_arr
        
        except Exception as e:
            logging.info("Exception occured while data transformation")
            
            raise CustomException(e,sys)
    
    
        
        
        
            
        

