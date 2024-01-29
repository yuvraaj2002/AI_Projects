import mlflow
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from Src.Exception import CustomException
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple

# experiment_tracker = Client().active_stack.experiment_tracker


# @step(experiment_tracker=experiment_tracker.name)
@step
def evaluation(
    model: RegressorMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Annotated[float, "R2_Score"], Annotated[float, "MAE"]]:
    """
    Args:
        model: RegressorMixin
        X_train: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_test: np.ndarray
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        # Getting predictions
        y_pred = model.predict(X_test)
        y_pred = np.expm1(y_pred)
        MAE_value = mean_absolute_error(y_test, y_pred)

        # K-fold cross-validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2")
        Avg_R2 = scores.mean()
        return Avg_R2, MAE_value

    except Exception as e:
        raise e
