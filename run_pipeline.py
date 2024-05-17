from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == '__main__':
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path='G:/Mi unidad/01_Oriol/00_Proyectos/MLOps_Customer_satisfaction/data/olist_customers_dataset.csv')
    



# zenml model-deployer register mlflow_tracker --flavor=mlflow
# zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set 
# mlflow ui --backend-store-uri "file:C:/Users/oriol/AppData/Roaming/zenml/local_stores/729a8a98-af13-47b4-b85e-04aeb0b82247/mlruns"

