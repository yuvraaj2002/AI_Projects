from Src.Exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,MinMaxScaler,FunctionTransformer
from Src.Utilities import save_object

from typing_extensions import Annotated
from typing import Tuple
from zenml import pipeline, step

class Data_Processing_Class:

    def __init__(self):
        pass

    def Process_data_method(self,Train_df,Test_df):
        """
        This method will take the training and testing dataframes, then it will use a pipeline to process the data
        and return 4 numpy arrays
        :param Train_df:
        :param Test_df:
        :return: X_train,X_test,y_train,y_test
        """
        try:

            X_train = Train_df.drop(['price'], axis=1)
            y_train = Train_df['price']
            X_test = Test_df.drop(['price'], axis=1)
            y_test = Test_df['price']

            # Defining the function transformer for doing log transformation
            trf = FunctionTransformer(func=np.log1p)

            X_train['built_up_area'] = trf.fit_transform(X_train['built_up_area'])
            X_test['built_up_area'] = trf.fit_transform(X_test['built_up_area'])
            y_train = trf.fit_transform(y_train)

            # Instantiating target encoder class
            Target_Encoder = ce.TargetEncoder()

            # Fit the TargetEncoder to the training data
            Target_Encoder.fit(X_train['sector'], y_train)

            # Transform the training data using the TargetEncoder
            X_train['sector'] = Target_Encoder.transform(X_train['sector'])
            X_test['sector'] = Target_Encoder.transform(X_test['sector'])

            # Column transformer for doing ordinal and nominal encoding
            encoding_transformer = ColumnTransformer(transformers=

                                                     [('Encode_PropT',
                                                       OneHotEncoder(handle_unknown='ignore', drop='first'),
                                                       [0]),

                                                      ('Encode_balcony',
                                                       OrdinalEncoder(categories=[['0', '1', '2', '3', '3+']],
                                                                      handle_unknown="use_encoded_value",
                                                                      unknown_value=np.nan), [4]),

                                                      ('Encode_AgeP', OrdinalEncoder(categories=[
                                                          ['Under Construction', 'New Property', 'Relatively New',
                                                           'Moderately Old', 'Old Property']],
                                                          handle_unknown="use_encoded_value",
                                                          unknown_value=np.nan), [5]),

                                                      ('Encode_Lux',
                                                       OrdinalEncoder(categories=[['Low', 'Medium', 'High']],
                                                                      handle_unknown="use_encoded_value",
                                                                      unknown_value=np.nan), [9]),

                                                      ('Encode_Floor',
                                                       OrdinalEncoder(
                                                           categories=[['Low Floor', 'Mid Floor', 'High Floor']],
                                                           handle_unknown="use_encoded_value",
                                                           unknown_value=np.nan), [10])],
                                                     remainder='passthrough')

            # Column transformer for doing feature scaling
            scaling_transformer = ColumnTransformer(transformers=[
                ('MinMaxScaling', MinMaxScaler(copy=False), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
                remainder='passthrough')

            # Stacking column transformers to create a Processing pipeline
            Processing_pipeline = Pipeline(steps=[('Encoding', encoding_transformer),
                                                  ('Scaling', scaling_transformer)], memory='Temp')

            # Processing the Training and testing data
            X_train = Processing_pipeline.fit_transform(X_train, y_train)
            X_test = Processing_pipeline.transform(X_test)

            # X_train = pd.DataFrame(X_train)
            # X_test = pd.DataFrame(X_test)
            return X_train,X_test,y_train.values,y_test.values

        except Exception as e:
            raise CustomException(e, sys)


@step
def process_data_step(Train_df:pd.DataFrame, Test_df:pd.DataFrame)-> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    try:
        process_data_obj = Data_Processing_Class()
        X_train, X_test, y_train, y_test = process_data_obj.Process_data_method(Train_df,Test_df)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)

