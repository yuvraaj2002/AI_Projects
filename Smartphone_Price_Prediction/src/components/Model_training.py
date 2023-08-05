from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class Model_training_Config:
    model_trained_path = os.path.join("artifacts", "Model.pkl")


class Train_Model:
    def __init__(self):
        self.model_file_path = Model_training_Config()

    def initialize_model_training(self,raw_data_path,pipe_path, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training Initialized")

            # Let's compare the best mode using pycaret
            df = pd.read_csv(raw_data_path)
            Input = df.drop(['price'], axis=1)
            Output = df['price']

            # Processing the input data using pipeline object
            with open(pipe_path, 'rb') as file:
                pipe = pickle.load(file)
            Input = pipe.transform(Input)
            logging.info("Complete data processed")

            # Instantiating pycaret
            pycaret_set = setup(data=Input, target=Output.values, train_size=0.8, preprocess=False)

            # Let's compare model and select best model
            best_model = compare_models(fold=5, cross_validation=True)
            model = best_model

            # Let's train the model
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Model predictions
            y_pred = model.predict(X_test)
            logging.info("Model predictions made")

            # Let's save the trained model
            save_object(file_path=self.model_file_path.model_trained_path, obj=model)
            logging.info("Saved model")

            # Let's calculate average R2 score using cross validation
            scores = cross_val_score(estimator=model, X=X_train, y=y_train, cv=50)
            return (np.mean(scores))

        except Exception as e:
            raise CustomException(e, sys)
