from Src.Logger import logging
from Src.Exception import CustomException
import os
import sys
import pandas as pd
from zenml import step
from Src.Process_data import Processing_Data
from Src.Train_model import Model_training


class Data_Ingestion:
    def __int__(self):
        pass

    def data_ingestion(self, path):
        try:
            logging.info("Initializing data ingestion")
            column_names = [
                "Sex",
                "Length",
                "Diameter",
                "Height",
                "Whole weight",
                "Shucked weight",
                "Viscera weight",
                "Shell weight",
                "Rings",
            ]
            df = pd.read_csv(path, header=None, names=column_names)
            logging.info("Loaded data successfully")

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
    ingest_obj = Data_Ingestion()
    raw_df = ingest_obj.data_ingestion(path)
    return raw_df


#
# if __name__ == "__main__":
#     data_ingestion_obj = Data_Ingestion()
#     df = data_ingestion_obj.data_ingestion("Dataset.txt")
#
#     process_data_obj = Processing_Data()
#     X_train, X_test, y_train, y_test = process_data_obj.process_data(df)
#
#     train_model_obj = Model_training()
#     train_model_obj.initialize_model_training(X_train, X_test, y_train, y_test)
