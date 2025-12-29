import os
import sys
import dill
import pickle

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
    


def evaluate_models(X_train, y_train, X_test, y_test, models,param):
    """
    This function takes a dictionary of models, trains them, 
    and returns a dictionary of their R2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i] # "list(models.values())"return a list with the values of models, and with
            # [i]==> we take in each iteration we take one Model from models. 
            
            para=param[list(models.keys())[i]]#===>check param in model_trainer.py==> para take the values of params, which are 
            #                                      criteria name with list of the criteria: {'criteria':[2,5,12]}


        
            gs = GridSearchCV(model,para,cv=3)  # Setup the Experiment:(no action here. we don't have the data)"Test this 'model' using these 'para' settings. Split data 3 times (cv=3)."
            gs.fit(X_train,y_train)  # Run the Experiment (This takes time!): It trains the model dozens of times with different settings.

           
            # Update the Model with the Winner
            # gs.best_params_ contains the winning settings (e.g., {'n_estimators': 128})
            # set_params applies them to our model variable.
            model.set_params(**gs.best_params_)
            '''
            --> gs.best_params_ returns a dictionary, like: {'n_estimators': 128, 'depth': 10}.
            --> The ** (Double Star) is Python unpacking. It converts that dictionary into arguments.
            --> It turns the code into this automatically:
            --> model.set_params(n_estimators=128, depth=10)
            '''

            model.fit(X_train,y_train) # Train the Final Model: Now that the model has the best settings, we train it one last time.

           
            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 Score for Test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score ## "list(models.keys())"return a list with the keys of report, and with
            # [i]==> we take fill in each iteration we fill the keys of the dict report  with the models scores as values. 
            

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)