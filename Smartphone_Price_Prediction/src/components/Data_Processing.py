from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    OrdinalEncoder,
    PowerTransformer,
    MinMaxScaler,
)
import category_encoders as ce
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class Data_Processing_Config:
    pipeline_path = os.path.join("artifacts", "Processing_pipeline.pkl")
    train_csv_path = os.path.join("artifacts", "train.csv")
    test_csv_path = os.path.join("artifacts", "test.csv")


class Data_processing:
    def __init__(self):
        self.paths = Data_Processing_Config()


    def initialize_data_processing(self, raw_data_path):
        try:
            logging.info("Data Processing started")

            # Let's first load the raw structurd data
            df = pd.read_csv(raw_data_path)

            # let's do train and test split and save the files
            train_data,test_data = train_test_split(df,random_state=1,train_size=0.8)

            # Let's make the directories to store the files
            os.makedirs(os.path.dirname(self.paths.train_csv_path), exist_ok=True)

            # Let's now store the files
            train_data.to_csv(self.paths.train_csv_path, index=False, header=True)
            test_data.to_csv(self.paths.test_csv_path, index=False, header=True)
            logging.info("Files saved")

            # Column transformer for univariate imputation (Mode)
            Mode_impute = ColumnTransformer(transformers=[
                ('Mode_imputation', SimpleImputer(strategy='most_frequent'), [0, 7, 8, 9])
            ], remainder='passthrough')

            # Column transformer for the ordinal encoding
            Oridnal_enc = ColumnTransformer(transformers=[
                ('OE_rating', OrdinalEncoder(categories=[['6+', '7+', '8+']], handle_unknown="use_encoded_value",
                                             unknown_value=np.nan), [0])]
                , remainder='passthrough')

            # Column transformer for Knn imputer
            Knn_imp = ColumnTransformer(transformers=[
                ('Knn_imputer', KNNImputer(n_neighbors=5, metric="nan_euclidean"), [0, 8, 9])
            ], remainder='passthrough')

            # Scaling the values
            scaling = ColumnTransformer(transformers=[
                ('Stand_scaling', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            ], remainder='passthrough')

            # Building a pipeline
            pipe = Pipeline(steps=[
                ('Mode_Imputation', Mode_impute),
                ('Ordinal_Encoding', Oridnal_enc),
                ('KNN_Imputer', Knn_imp),
                ('Scaling', scaling)
            ])
            logging.info("Created a pipeline successfully")

            # Seperating the dependent and independent variable
            X_train = train_data.drop(['price'], axis=1)
            y_train = train_data['price']

            X_test = test_data.drop(['price'], axis=1)
            y_test = test_data['price']
            logging.info("Created X_train,y_train and X_test,y_test successfully")

            #Let's now process the data
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)
            logging.info("Train and test data processed")

            y_train = np.array(y_train.values)
            y_test = np.array(y_test.values)

            #Let's now save the pipeline
            save_object(file_path=self.paths.pipeline_path, obj=pipe)
            logging.info("Saved pipeilne object")

            logging.info("Data Processing completed")
            return (X_train, y_train, X_test, y_test,self.paths.pipeline_path)

        except Exception as e:
            raise CustomException(e, sys)