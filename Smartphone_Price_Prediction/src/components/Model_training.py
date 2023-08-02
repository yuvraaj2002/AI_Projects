from src.logger import logging
from src.exception import CustomException
import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

    def initialize_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training Initialized")

            model = RandomForestRegressor()

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
