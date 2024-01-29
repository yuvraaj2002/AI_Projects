import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_absolute_error


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning using the hyperopt library
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def objective(self, space):
        model = XGBRegressor(
            n_estimators=int(space["n_estimators"]),
            max_depth=int(space["max_depth"]),
            gamma=space["gamma"],
            reg_alpha=space["reg_alpha"],
            reg_lambda=space["reg_lambda"],
            colsample_bytree=space["colsample_bytree"],
            min_child_weight=int(space["min_child_weight"]),
            random_state=space["seed"],
        )

        # Training the model
        model.fit(self.X_train, self.y_train)

        # Getting predictions from the trained model and computing mean absolute error
        y_pred = model.predict(self.X_test)
        y_pred = np.expm1(y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)

        # Return the value to minimize (in this case, MAE)
        return {"loss": mae, "status": STATUS_OK}

    def optimize(self):
        # Defining the domain space
        space = {
            "max_depth": hp.quniform("max_depth", 3, 18, 1),
            "gamma": hp.uniform("gamma", 0, 5),
            "reg_alpha": hp.uniform("reg_alpha", 0, 1),
            "reg_lambda": hp.uniform("reg_lambda", 0, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
            "n_estimators": 180,
            "seed": 0,
        }

        trials = Trials()
        best_hyperparams = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )
        return best_hyperparams
