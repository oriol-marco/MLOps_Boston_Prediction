import logging
import pandas as pd
from zenml import step
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple
import mlflow

from zenml.client import Client


experimet_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experimet_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "R2 Score"],
    Annotated[float, "rmse"],
]:
    """               
    Evaluates the model on the given data.
    
    Args:
        model (RegressorMixin): The model to evaluate.
        X_test (pd.DataFrame): The features to evaluate the model on.
        y_test (pd.DataFrame): The target to evaluate the model on.
    """
    try: 
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)
        
        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2_score", r2_score)
        
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        
        return r2_score, rmse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e