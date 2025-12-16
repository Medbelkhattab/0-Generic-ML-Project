import os
import sys
import dill

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV # (Optional if he does hyperparameter tuning later, but good to have)

from src.exception import CustomException

#save_object function: Its job is to take a Python object  (ColumnTransformer) (#obj= preprocessor_obj =self.get_data_transformer_object())
# and save it to the hard drive as a file (usually .pkl)(file_path=self.data_transformation_config.preprocessor_obj_file_path)

def save_object(file_path, obj): 
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj) #dill: It is a Python library used for Serialization. It converts Python objects in
                                    #memory (RAM) into a byte stream (file) so they can be stored or transferred.

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    This function takes a dictionary of models, trains them, 
    and returns a dictionary of their R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i] # "list(models.values())"return a list with the values of models, and with
            # [i]==> we take in each iteration we take one Model from models. 
            
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 Score for Test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score ## "list(models.keys())"return a list with the keys of report, and with
            # [i]==> we take fill in each iteration we fill the keys of the dict report  with the models scores as values. 
            

        return report

    except Exception as e:
        raise CustomException(e, sys)