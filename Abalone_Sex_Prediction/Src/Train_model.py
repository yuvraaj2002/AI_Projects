from Src.Logger import logging
from Src.Exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
import pickle
from Src.Utils import save_object
from Src.Model_Config import ModelNameConfig
from sklearn.base import ClassifierMixin
from zenml.steps import step
import mlflow
from Src.Model_definition import (
    HyperparameterTuner,
    LogisticRegressionModel
)

# experiment_tracker = Client().active_stack.experiment_tracker


class Model_training:
    def __init__(self):
        pass

    def initialize_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Model training Initialized")

            model = None
            tuner = None

            # Based on model configuration we will instantiate the model
            model_config_obj = ModelNameConfig()

            if model_config_obj.model_name == "Logistic Regression":
                # mlflow.lightgbm.autolog()
                model = LogisticRegressionModel()
            else:
                raise ValueError("Model name not supported")

            # Instantiating HyperparameterTuner
            tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

            # If fine_tuning is set to be true in Model configuration file
            if model_config_obj.fine_tuning == True:
                best_params = tuner.optimize()  # Tuning the parameters
                trained_model = model.train(X_train, y_train, **best_params)  # Training model with new tuned params

                # Let's save the trained model
                save_object(file_path=model_config_obj.model_storage_path, obj=trained_model)
                logging.info("Saved tuned and trained model")
            else:
                trained_model = model.train(X_train, y_train)  # Train model with base params
                save_object(file_path=model_config_obj.model_storage_path, obj=trained_model)
                logging.info("Saved tuned and trained model")

            return trained_model

        except Exception as e:
            raise CustomException(e, sys)


# @step(experiment_tracker=experiment_tracker.name)
@step
def train_model(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array
) -> ClassifierMixin:
    """
    Args:
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        y_test: np.array
    Returns:
        model: ClassifierMixin
    """
    model_train_obj = Model_training()
    trained_model = model_train_obj.initialize_model_training(
        X_train, X_test, y_train, y_test
    )
    return trained_model
