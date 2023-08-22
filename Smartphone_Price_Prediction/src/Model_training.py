from src.logger import logging
from src.exception import CustomException
import sys
import numpy as np
from src.utils import save_object
from Model_info.Model_config import ModelNameConfig
from sklearn.base import RegressorMixin
from zenml.steps import step
import mlflow
from Model_info.Models_definition import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
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

            # Based on model name mentioned in the model configuration we will instantiate the model
            model_config_obj = ModelNameConfig()

            if model_config_obj.model_name == "lightgbm":
                mlflow.lightgbm.autolog()
                model = LightGBMModel()
            elif model_config_obj.model_name == "randomforest":
                mlflow.sklearn.autolog()
                model = RandomForestModel()
            elif model_config_obj.model_name == "xgboost":
                mlflow.xgboost.autolog()
                model = XGBoostModel()
            elif model_config_obj.model_name == "linear_regression":
                mlflow.sklearn.autolog()
                model = LinearRegressionModel()
            else:
                raise ValueError("Model name not supported")

            # Instantiating HyperparameterTuner
            tuner = HyperparameterTuner(model, X_train, y_train, X_test, y_test)

            # If fine_tuning is set to be true in Model configuration file
            if model_config_obj.fine_tuning:
                best_params = tuner.optimize()  # Tuning the parameters
                trained_model = model.train(
                    X_train, y_train, **best_params
                )  # Training model with new tuned params

                # Let's save the trained model
                save_object(file_path=model_config_obj.model_storage_path, obj=trained_model)
                logging.info("Saved tuned and trained model")
            else:
                trained_model = model.train(
                    X_train, y_train
                )  # Train model with base params
                save_object(file_path=model_config_obj.model_storage_path, obj=trained_model)
                logging.info("Saved model trained with base params")

            return trained_model

        except Exception as e:
            raise CustomException(e, sys)


# @step(experiment_tracker=experiment_tracker.name)
@step
def train_model(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> RegressorMixin:
    """
    Args:
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    Returns:
        model: RegressorMixin
    """
    model_train_obj = Model_training()
    trained_model = model_train_obj.initialize_model_training(
        X_train, X_test, y_train, y_test
    )
    return trained_model