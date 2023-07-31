from src.logger import logging
from src.exception import CustomException
import sys
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class Model_training_Config:
    model_trained_path = os.path.join("artifacts", "model.pkl")


class Train_Model:
    def __init__(self):
        self.model_file_path = Model_training_Config()

    def initialize_model_training(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training intialized")

            model = GradientBoostingClassifier(
                loss="log_loss",
                learning_rate=0.11818576935196282,
                n_estimators=322,
                subsample=0.919894055658876,
                criterion="friedman_mse",
                min_samples_split=10,
                min_samples_leaf=5,
                min_weight_fraction_leaf=0.020616418092142458,
                max_depth=9,
                min_impurity_decrease=0.8342382108843562,
            )

            # Let's train the model
            model.fit(X_train, y_train)
            logging.info("Model training completed")

            # Model predictions
            y_pred = model.predict(X_test)
            logging.info("Model predictions made")

            # Let's save the trained model
            save_object(file_path=self.model_file_path.model_trained_path, obj=model)
            logging.info("Saved model")

            # Let's calculate accuracy, precision, recall and F1-Score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")
            logging.info("Model training completed")

            return (accuracy, precision, recall, f1)

        except Exception as e:
            raise CustomException(e, sys)
