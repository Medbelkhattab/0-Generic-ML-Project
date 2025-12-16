import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging   

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        the role of this function is data transformation
        '''
        try:
            numericals_columns=['writing_score','reading_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education',
                              'lunch', 'test_preparation_course']
            
            #num_pipeline & cat_pipeline= x = what to do without knowing to who.

            num_pipeline=Pipeline(    # A wrapper to execute the following steps in order
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), # 1. Handle Missing Data: Replaces 
                                                                    #NaNs with the 'Median' (middle value)
                    ("scaler",StandardScaler())  # 2. Scale Numbers: Adjusts values so Mean=0 and Variance=1
                ]
            )
            logging.info("Numerical standard scaling cempleted")  

            cat_Pipeline=Pipeline(  # A wrapper for the categorical data steps
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),# 1. Handle Missing Data: Replaces 
                                                                    #NaNs with the value that appears most often
                ("one_hot_encoder",OneHotEncoder()),# 2. Convert Text to Math: Turns 
                                                        # categories (e.g., "Red") into binary columns (0s and 1s)
                ("scalar",StandardScaler(with_mean=False))# 3. Scale Result: Normalizes the new binary data
                ]           
            )
            logging.info(f"Numericals columns:{numericals_columns}") 
            logging.info(f"Categorical columns:{categorical_columns}") 

            #ColumnTransformer()= a class that define x to who...
            #it splits the data given as argument to 
            # fit_transform(input_feature_train_df) between categorical and numerical to apply to each type the 
            # right  calculation.

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline",num_pipeline,numericals_columns),
                ("cat_Pipeline",cat_Pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
#fit = based on the class ColumnTransformer() it learns the specific statistical 
# parameters (mean, std dev, unique categories)
#  from the data, defining how the math will be done.

#transform= it excutes the actual math and compute.
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numericals_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                ]
            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path, #= os.join('artifacts','preprocessor.pkl')
                obj=preprocessing_obj

            
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
                            






        except Exception as e:
            raise CustomException(e,sys)
            

                    
         

    