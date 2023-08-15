import logging
from abc import ABC, abstractmethod
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            X_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Target data
            X_test: Testing data
            y_test: Testing target
        """
        pass



class LogisticRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        lr = LogisticRegression(**kwargs)
        lr.fit(X_train, y_train)
        return lr

    # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, X_train, y_train, X_test, y_test):
        lr = self.train(X_train, y_train)
        return lr.score(X_test, y_test)


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.x_train = X_train
        self.y_train = y_train
        self.x_test = X_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.model.optimize(
                trial, self.X_train, self.y_train, self.X_test, self.y_test
            ),
            n_trials=n_trials,
        )
        return study.best_trial.params
