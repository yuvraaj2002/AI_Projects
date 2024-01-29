from Src.Exception import CustomException
import sys
import numpy as np
from Src.Utilities import save_object
from Src.Model_Info.Model_Configuration import ModelNameConfig
from Src.Model_Info.Model_Tuning import HyperparameterTuner
from sklearn.base import RegressorMixin
from zenml import step
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# from Src.Model_definition import (
#     HyperparameterTuner,
#     LogisticRegressionModel
# )

# experiment_tracker = Client().active_stack.experiment_tracker


class Model_training_Class:
    def __init__(self):
        pass

    def initialize_model_training(self, X_train, X_test, y_train, y_test):
        try:
            model = None
            tuner = None

            # Based on model configuration we will instantiate the model
            model_config_obj = ModelNameConfig()

            if model_config_obj.model_name == "Xgboost":
                # mlflow.lightgbm.autolog()
                model = XGBRegressor()
            elif model_config_obj.model_name == "Extra Trees":
                model = ExtraTreesRegressor()
            elif model_config_obj.model_name == "Random Forest":
                model = RandomForestRegressor()
            else:
                raise ValueError("Model name not supported")

            # If fine_tuning is set to be true in Model configuration file
            if model_config_obj.fine_tuning == True:
                find_hparms_obj = HyperparameterTuner(X_train, X_test, y_train, y_test)
                hyper_parms = (
                    find_hparms_obj.optimize()
                )  # Getting best set of hyper-parameters

                # Create the XGBoost Regressor with the best hyperparameters
                Tuned_model = XGBRegressor(
                    max_depth=int(hyper_parms["max_depth"]),
                    gamma=hyper_parms["gamma"],
                    reg_alpha=hyper_parms["reg_alpha"],
                    reg_lambda=hyper_parms["reg_lambda"],
                    colsample_bytree=hyper_parms["colsample_bytree"],
                    min_child_weight=hyper_parms["min_child_weight"],
                    n_estimators=180,
                    random_state=0,
                )

                # Training the model
                Tuned_model.fit(X_train, y_train)

                # Let's save the trained model
                save_object(
                    file_path=model_config_obj.model_storage_path, obj=Tuned_model
                )
                return Tuned_model

            else:
                Without_Tuning_model = model.fit(
                    X_train, y_train
                )  # Train model with base params
                save_object(
                    file_path=model_config_obj.model_storage_path,
                    obj=Without_Tuning_model,
                )
                return Without_Tuning_model

        except Exception as e:
            raise CustomException(e, sys)


# @step(experiment_tracker=experiment_tracker.name)
@step
def train_model(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray
) -> RegressorMixin:
    """
    Args:
        X_train: np.array,
        X_test: np.array,
        y_train: np.array,
        y_test: np.array
    Returns:
        model: RegressorMixin
    """
    model_train_obj = Model_training_Class()
    trained_model = model_train_obj.initialize_model_training(
        X_train, X_test, y_train, y_test
    )
    return trained_model
