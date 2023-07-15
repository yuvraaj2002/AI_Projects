from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
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
    pipeline_path = os.path.join("artifacts", "Processing_pipe.pkl")


class Data_processing:
    def __init__(self):
        self.pipe_path = Data_Processing_Config()

    def initialize_data_processing(self, train_data_path, test_data_path):
        try:
            logging.info("Data Processing started")

            # Let's load the training and testing data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Training and testing data loaded")

            # Let's first remove some unnecessary columns
            train_data.drop(["enrollee_id", "city"], axis=1, inplace=True)
            test_data.drop(["enrollee_id", "city"], axis=1, inplace=True)
            logging.info("Removed unnecessary columns")

            # Let's remove any duplicates
            train_data.drop_duplicates(inplace=True)
            test_data.drop_duplicates(inplace=True)
            logging.info("Removed duplicates")

            # Let's make X_train,y_train and X_test,y_test
            X_train = train_data.drop(["target"], axis=1)
            y_train = train_data["target"]
            X_test = test_data.drop(["target"], axis=1)
            y_test = test_data["target"]
            logging.info("Created X_train,y_train and X_test,y_test")

            # Let's now build a pipeline
            # Define the column transformer for imputation
            Simple_impute_transformer = ColumnTransformer(
                transformers=[
                    ("mean_imputer", SimpleImputer(strategy="mean"), [0, 6, 9]),
                    ("mode_imputer", SimpleImputer(strategy="most_frequent"), [3, 4]),
                ],
                remainder="passthrough",
            )

            # Define the column transformer for encoding
            encode_values = ColumnTransformer(
                transformers=[
                    (
                        "Encode_ordinal_Re",
                        OrdinalEncoder(
                            categories=[
                                ["No relevent experience", "Has relevent experience"]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [6],
                    ),
                    (
                        "Encode_ordinal_eu",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "no_enrollment",
                                    "Part time course",
                                    "Full time course",
                                ]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [3],
                    ),
                    (
                        "Encode_ordinal_el",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "Primary School",
                                    "High School",
                                    "Graduate",
                                    "Masters",
                                    "Phd",
                                ]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [4],
                    ),
                    (
                        "Encode_ordinal_cs",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "<10",
                                    "10/49",
                                    "50-99",
                                    "100-500",
                                    "500-999",
                                    "1000-4999",
                                    "5000-9999",
                                    "10000+",
                                ]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [8],
                    ),
                    (
                        "Encode_target_major",
                        ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
                        [7],
                    ),
                    (
                        "Encode_target_ct",
                        ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
                        [9],
                    ),
                    (
                        "Encode_target_gen",
                        ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
                        [5],
                    ),
                ],
                remainder="passthrough",
            )

            Knn_imputer = ColumnTransformer(
                transformers=[
                    (
                        "Knn_Imputer",
                        KNNImputer(n_neighbors=5, metric="nan_euclidean"),
                        [3, 4, 5, 6],
                    )
                ],
                remainder="passthrough",
            )

            yeo_transformation = ColumnTransformer(
                transformers=[("Yeo-Johnson", PowerTransformer(), [9])],
                remainder="passthrough",
            )

            # Column transformer to do feature scaling
            scaling_transformer = ColumnTransformer(
                transformers=[("scale_transformer", MinMaxScaler(), [1])],
                remainder="passthrough",
            )

            # Define the final pipeline
            pipe = Pipeline(
                steps=[
                    ("impute_transformer", Simple_impute_transformer),
                    ("encode_values", encode_values),
                    ("Knn_imputer", Knn_imputer),
                    ("Yeo-Johnson-Transformation", yeo_transformation),
                    ("Scaling", scaling_transformer),
                ]
            )

            # Let's train and test data
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)
            logging.info("Train and test data processed")

            y_train = np.array(y_train.values)
            y_test = np.array(y_test.values)

            # Let's now save the pipeline
            save_object(file_path=self.pipe_path.pipeline_path, obj=pipe)
            logging.info("Saved pipeilne object")

            logging.info("Data Processing completed")
            return (X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)
