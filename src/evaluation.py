import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our model

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Abstract method to calculate scores

        Args:
            y_true (np.ndarray)
            y_pred (np.ndarray)

        Returns:
            dict: _description_
        """
        pass
    
class MSE(Evaluation):
    """
    Evaluation strategy using Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate scores using Mean Squared Error

        Args:
            y_true (np.ndarray)
            y_pred (np.ndarray)
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE calculated")
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
        
class R2(Evaluation):
    """
    Evaluation strategy using R2
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate scores using R2

        Args:
            y_true (np.ndarray)
            y_pred (np.ndarray)
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation strategy using Root Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate scores using Root Mean Squared Error

        Args:
            y_true (np.ndarray)
            y_pred (np.ndarray)
        """
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
        
