import os
import sys
import pandas as pd
import numpy as np
from zenml import step
from Src.Exception import CustomException
from typing_extensions import Annotated
from typing import Tuple
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler,
    FunctionTransformer,
)
from sklearn.model_selection import train_test_split


class DIP:
    def __int__(self):
        pass

    def Data_Ingest_Process(self, path):
        """
        This method will take the file path as input and will load the data from the given path,
        perform train, test split and save the files in the specified directory
        :param path: file path
        :return: train and testing dataframe
        """
        try:
            # Reading the data from the csv file
            df = pd.read_csv(path)

            # Performing the train test split
            Train_df, Test_df = train_test_split(df, train_size=0.8, shuffle=True)

            # Let's make the directory to store the files
            os.makedirs(
                os.path.dirname(os.path.join("Artifacts", "Train.csv")), exist_ok=True
            )
            os.makedirs(
                os.path.dirname(os.path.join("Artifacts", "Test.csv")), exist_ok=True
            )

            # Let's now store the files
            Train_df.to_csv(
                os.path.join("Artifacts", "Train.csv"), index=False, header=True
            )
            Test_df.to_csv(
                os.path.join("Artifacts", "Test.csv"), index=False, header=True
            )

            return Train_df, Test_df

        except Exception as e:
            raise CustomException(e, sys)


@step
def ingest_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ingest_obj = DIP()
    Train_df, Test_df = ingest_obj.Data_Ingest_Process(path)
    return Train_df, Test_df


# USE THE BELOW-MENTIONED CODE TO CHECK IF EVERYTHING IS WORKING FINE
# if __name__ == "__main__":
#     ingest_obj = DIP()
#     Train_df, Test_df = ingest_obj.Data_Ingest_Process("/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Combined_Cleandata_V4.csv")
#     print(Train_df.head(5))
