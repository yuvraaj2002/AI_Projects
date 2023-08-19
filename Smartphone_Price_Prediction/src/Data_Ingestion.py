import sys
import os
import pandas as pd
from zenml.steps import step
from src.logger import logging
from src.exception import CustomException


class Data_Ingestion:
    def __init__(self):
        pass

    def initialize_data_ingestion(self, path):
        """
        This method will take the path where the data is stored, and it will first store
        the raw data in the specified directory and will return us the raw_df.
        Here for simplicity we are simply using the path of csv file but here we can use any
        data source.
        :param path: Path of the raw data file
        :return: raw uncleaned and unprocessed data frame
        """
        try:
            logging.info("Initializing data ingestion")

            # Reading the data from the csv file
            df = pd.read_csv(path)
            logging.info("Successfully loaded the data from csv file")

            # Let's make the directories to store the files
            os.makedirs(
                os.path.dirname(os.path.join("Artifacts", "Raw.csv")), exist_ok=True
            )

            # Let's now store the files
            df.to_csv(os.path.join("Artifacts", "Raw.csv"), index=False, header=True)
            logging.info("CSV files saved successfully")

            logging.info("Data Ingestion completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)


@step
def ingest_data(path: str) -> pd.DataFrame:
    try:
        # Instantiating the Data_Ingestion class
        ingest_data_obj = Data_Ingestion()
        df = ingest_data_obj.initialize_data_ingestion(path)
        return df
    except Exception as e:
        raise CustomException(e, sys)
