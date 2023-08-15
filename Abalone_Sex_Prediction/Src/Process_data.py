from Src.Logger import logging
from Src.Exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from typing_extensions import Annotated
from typing import Tuple
from Src.Utils import save_object
from zenml.steps import step


class Processing_Data:
    def __int__(self):
        pass

    def process_data(self, raw_df):
        try:
            logging.info("Initializing data processing")

            # Separating the data in dependent and independent variables
            X = raw_df.drop(["Sex"], axis=1)
            y = raw_df["Sex"]

            # Train test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=1
            )

            # Scaling the values
            scaling = ColumnTransformer(
                transformers=[
                    ("Stand_scaling", StandardScaler(), [0, 1, 2, 3, 4, 5, 6, 7])
                ],
                remainder="passthrough",
            )

            # Building a pipeline
            pipe = Pipeline(steps=[("Scaling", scaling)])
            logging.info("Created data processing pipeline")

            # Process the data using pipeline
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)

            # Instantiating label encoder class
            Le = LabelEncoder()

            y_train = Le.fit_transform(y_train)
            y_test = Le.transform(y_test)
            logging.info("Processed the training and testing data successfully")

            # Saving the pipeline
            save_object(os.path.join("Artifacts", "Processing_pipeline.pkl"), pipe)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)


@step
def process_data(
    raw_df: pd.DataFrame,
) -> Tuple[
    Annotated[np.array, "X_train"],
    Annotated[np.array, "X_test"],
    Annotated[np.array, "y_train"],
    Annotated[np.array, "y_test"],
]:
    try:
        process_data_obj = Processing_Data()
        X_train, X_test, y_train, y_test = process_data_obj.process_data(raw_df)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)
