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
from Src.Evaluation_parms import accuracy, precision, recall, f1
from typing import Tuple

# experiment_tracker = Client().active_stack.experiment_tracker


# @step(experiment_tracker=experiment_tracker.name)
@step
def evaluation(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Annotated[float, "acc_value"], Annotated[float, "precision_value"],
Annotated[float, "recall_value"], Annotated[float, "f1_value"]]:
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

        # Using the accuracy class
        acc_class = accuracy()
        acc_value = acc_class.calculate_score(y_test, prediction)
        mlflow.log_metric("accuracy_value", acc_value)

        # Using the precision
        precision_class = precision()
        precision_value = precision_class.calculate_score(y_test, prediction)
        mlflow.log_metric("precision_value", precision_value)

        # Using the recall class
        recall_class = recall()
        recall_value = recall_class.calculate_score(y_test, prediction)
        mlflow.log_metric("recall_value", recall_value)

        # Using the f1 class
        f1_class = f1()
        f1_value = f1_class.calculate_score(y_test, prediction)
        mlflow.log_metric("f1_value", f1_value)

        return acc_value, precision_value, recall_value, f1_value
    except Exception as e:
        logging.error(e)
        raise e


