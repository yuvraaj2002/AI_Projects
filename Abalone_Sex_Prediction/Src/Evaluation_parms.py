import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class accuracy(Evaluation):
    """
    Evaluation strategy that uses accuracy score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse: float
        """
        try:
            logging.info("Entered the calculate_score method of the Accuracy class")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info("The accuracy value is: " + str(accuracy))
            return accuracy
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the accuracy class. Exception message:  "
                + str(e)
            )
            raise e


class precision(Evaluation):
    """
    Evaluation strategy that uses precision score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            precision = precision_score(y_true, y_pred,average='weighted')
            logging.info("The precision value is: " + str(precision))
            return precision
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the precision class. Exception message:  "
                + str(e)
            )
            raise e


class recall(Evaluation):
    """
    Evaluation strategy that uses recall score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            recall = recall_score(y_true, y_pred,average='weighted')
            logging.info("The recall value is: " + str(recall))
            return recall
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the recall class. Exception message:  "
                + str(e)
            )
            raise e



class f1(Evaluation):
    """
    Evaluation strategy that uses f1 score
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info("Entered the calculate_score method of the f1 class")
            f1_value = f1_score(y_true, y_pred,average='weighted')
            logging.info("The f1 score value is: " + str(f1_value))
            return f1_value
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the f1_value class. Exception message:  "
                + str(e)
            )
            raise e

