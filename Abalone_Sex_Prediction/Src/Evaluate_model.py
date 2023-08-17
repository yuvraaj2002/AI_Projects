import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from Src.Logger import logging
from Src.Exception import CustomException
from Src.Evaluation_parms import RMSE,R2Score,MSE
from typing import Tuple

#experiment_tracker = Client().active_stack.experiment_tracker


# @step(experiment_tracker=experiment_tracker.name)
@step
def evaluation(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Args:
        model: ClassifierMixin
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
        logging.error(e)
        raise e


