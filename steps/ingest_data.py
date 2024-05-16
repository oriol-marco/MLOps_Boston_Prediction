import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    A class for ingesting data from a specified path.
    
    Args:
        data_path (str): The path to the data file.
    """
    def __init__(self, data_path: str):
        """Initializes the IngestData class.

        Args:
            data_path (str): The path to the data file.
        """
        self.data_path = data_path
        
    def get_data(self):
        """Ingests the data from the specified path.
        
        Returns:
            pandas.DataFrame: The ingested data.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a given path.

    Args:
        data_path (str): path to the data file
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        return None
    