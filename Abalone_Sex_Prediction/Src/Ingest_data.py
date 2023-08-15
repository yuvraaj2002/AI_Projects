from Src.Logger import logging
from Src.Exception import CustomException
import os
import sys
import pandas as pd
from zenml.steps import step


class Data_Ingestion:

    def data_ingestion(self,path):
        try:
            logging.info("Initializing data ingestion")
            df = pd.read_csv(path)

            # Let's make the directories to store the files
            os.makedirs(os.path.dirname(os.path.join("Artifacts", "Raw.csv")), exist_ok=True)

            # Let's now store the files
            df.to_csv(os.path.join("Artifacts", "Raw.csv"), index=False, header=True)
            logging.info("CSV files saved successfully")

            logging.info("Data Ingestion completed")
            return df

        except Exception as e:
            raise CustomException(e,sys)


@step
def ingest_data(path:str) -> pd.DataFrame:

    ingest_obj = Data_Ingestion()
    raw_df = ingest_obj.data_ingestion("adfa")
    return raw_df

