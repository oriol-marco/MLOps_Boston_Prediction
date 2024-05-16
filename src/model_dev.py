import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for models strategies.
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train models

        Args:
            X_train (_type_): Training data.
            y_train (_type_): Traing labels.
            
        Returns: 
            None
        """
class LinearRegressionModel(Model):
    """
    Linear Regression Model.
    """
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train Linear Regression model

        Args:
            X_train: Training data.
            y_train: Traing labels.
            
        Returns: 
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training successful.")
            return reg
        except Exception as e:
            logging.error("Error in model training: {}".format(e))
            raise e