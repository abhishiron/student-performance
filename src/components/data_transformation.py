import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing object
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch',
                'test_preparation_course'
            ]
            
            # Pipeline for numerical features
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  # handles missing values
                ('scaler', StandardScaler())  # standardize the features
            ])
            
            # Pipeline for categorical features
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),  # convert categorical to numerical
                ('scaler', StandardScaler(with_mean=False))  # scale sparse matrices
            ])
            
            logging.info("Numerical transformers created successfully.")
            logging.info("Categorical transformers created successfully.")
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_columns),
                    ('cat', categorical_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        This function initiates the data transformation
        """
        try:
            logging.info("Initiating data transformation...")
            
            # Read train and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("Read train and test data successfully.")
            
            # Get preprocessing object
            preprocessor = self.get_data_transformer_object()
            
            target_column = "math_score"
            
            # Separate features and target variable
            input_feature_train_df = train_data.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_data[target_column]
            
            input_feature_test_df = test_data.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_data[target_column]
            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            # Transform the input features
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Combine transformed features with target variable
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]
            
            logging.info("Saving the preprocessor object.")
            
            # Save the preprocessor object
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            
            logging.info("Data transformation completed successfully.")
            
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error("Error occurred during data transformation.")
            raise CustomException(e, sys)