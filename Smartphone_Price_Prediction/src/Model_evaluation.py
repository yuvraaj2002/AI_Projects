import logging
from src.logger import logging
from src.exception import CustomException
import mlflow
import os
import numpy as np
import pandas as pd
import sys
from src.Evaluation_definition import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml.steps import step

# from zenml.client import Client
# experiment_tracker = Client().active_stack.experiment_tracker
from typing import Tuple


# @step(experiment_tracker=experiment_tracker.name)
@step
def evaluation(
    model: RegressorMixin, X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Args:
        model: RegressorMixin
        X_test: np.ndarray
        y_test: np.ndarray
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)

        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2_score, rmse
    except Exception as e:
        raise CustomException(e, sys)
