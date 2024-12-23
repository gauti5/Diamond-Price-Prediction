import sys
import os
from pathlib import Path
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@dataclass
class ModelTrainerConfig():
    trained_model_file_path=os.path.join('Artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_training(self,train_array, test_array):
        try:
            
            logging.info("Splitting the training and testing data")
            
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Gradient Boost Regressor": GradientBoostingRegressor()
            }
            
            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            print(model_report)
            
            print("\n=================================================================\n")
            
            logging.info(f"model report : {model_report}")
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best Model Found for both training and testing data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        
        except Exception as e:
            logging.info("Exception Occured while training the model")
            raise CustomException(e,sys)
        
        
    
        