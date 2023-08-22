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
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
import category_encoders as ce
from dataclasses import dataclass
from src.utils import save_object
from typing_extensions import Annotated
from typing import Tuple
from zenml.steps import step, Output


@dataclass
class Data_Processing_Config:
    pipeline_path = os.path.join("Artifacts", "Processing_pipeline.pkl")
    train_csv_path = os.path.join("Artifacts", "train.csv")
    test_csv_path = os.path.join("Artifacts", "test.csv")


class feature_engineering:
    def __int__(self):
        pass

    def extract_encode_sim(self, data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        # Empty lists to store data
        Has_5g = []
        Add_Features = []

        for item in data["sim"]:
            Has_5g.append(1 if "5G," in item.split() else 0)
            Add_Features.append(
                1 if "NFC," in item.split() or "IR," in item.split(",") else 0
            )

        Has_5g = pd.Series(Has_5g)
        Add_Features = pd.Series(Add_Features)
        return [Has_5g, Add_Features]

    def extract_encode_ram(self, data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        RAM = []
        Storage = []

        for item in data["ram"]:
            contents = item.split()
            if ("RAM," in contents) or (
                "inbuilt" in contents
            ):  # Filtered out any wrong info
                # Now we just need to figure out if both are present or only one is present
                if ("RAM," in contents) and ("inbuilt" in contents):
                    RAM.append(int(contents[0]))
                    Storage.append(int(contents[3]))
                elif "RAM," in contents:
                    RAM.append(int(contents[0]))
                    Storage.append(np.nan)
                elif "inbuilt" in contents:
                    RAM.append(np.nan)
                    Storage.append(int(contents[0]))

            else:
                RAM.append(np.nan)
                Storage.append(np.nan)

        RAM = pd.Series(RAM)
        Storage = pd.Series(Storage)
        return [RAM, Storage]

    def extract_encode_battery(self, data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        Battery = []
        Charging = []

        for item in data["battery"]:
            contents = item.split()
            if ("Battery" in contents) or (
                "Charging" in contents
            ):  # Filtered out any wrong info
                if (
                    ("Battery" in contents)
                    and ("Charging" in contents)
                    and len(contents) == 7
                ):  # Means the Watt is given
                    Battery.append(float(contents[0]))
                    if float(contents[4][:-1]) < 50.0:
                        Charging.append(int(1))
                    elif (float(contents[4][:-1]) > 50.0) and (
                        float(contents[4][:-1]) < 100.0
                    ):
                        Charging.append(int(2))
                    else:
                        Charging.append(int(3))

                # Means keywords are present but value of fast charging is not mentioned
                elif (
                    ("Battery" in contents)
                    and ("Charging" in contents)
                    and len(contents) == 6
                ):
                    Battery.append(float(contents[0]))
                    Charging.append(np.nan)

                elif "Battery" in contents:
                    Battery.append(float(contents[0]))
                    Charging.append(np.nan)

            else:
                Battery.append(np.nan)
                Charging.append(np.nan)

        Battery = pd.Series(Battery)
        Charging = pd.Series(Charging)
        return [Battery, Charging]

    def extract_encode_display(self, data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        PPI = []
        Screen_RR = []

        for item in data["display"]:
            contents = item.split()
            if ("inches," in contents) and ("Display"):  # Filtered out any wrong info
                diagonal_len = float(contents[0])
                px1_val = float(contents[2])
                px2_val = float(contents[4])
                total_px_value = px1_val * px2_val
                PPI.append(int(total_px_value / diagonal_len))
            else:
                PPI.append(np.nan)

        for item in data["display"]:
            contents = item.split()
            if "Hz" in contents:
                Screen_RR.append(int(contents[6]))
            else:
                Screen_RR.append(np.nan)

        PPI = pd.Series(PPI)
        Screen_RR = pd.Series(Screen_RR)
        return [Screen_RR, PPI]

    def extract_encode_camera(self, data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        rear_cams = []
        Total_fmp = []

        data["camera"].fillna(data["camera"].mode()[0], inplace=True)

        for item in data["camera"]:
            contents = item.split()
            if (
                ("Rear" in contents) and ("Front" in contents) and ("&" in contents)
            ):  # Filtered out any wrong info
                if "Quad" in contents:
                    rear_cams.append(4)
                elif "Triple" in contents:
                    rear_cams.append(3)
                elif "Dual" in contents:
                    rear_cams.append(2)
                else:
                    rear_cams.append(1)

                Total_fmp.append(float(item.split("&")[-1].split()[0]))

            else:
                rear_cams.append(np.nan)
                Total_fmp.append(np.nan)

        rear_cams = pd.Series(rear_cams)
        Total_fmp = pd.Series(Total_fmp)
        return [rear_cams, Total_fmp]

    def extract_processor_info(self, data):
        Processor_name = []
        Processor_core = []
        Processor_GHz = []

        for item in data["processor"]:
            contents = item.split()
            if "Processor" in contents:  # Filtered out wrong info
                if ("Core," in contents) and (
                    "GHz" in contents
                ):  # When all name,core and Ghz are present
                    Processor_name.append(contents[0])
                    Processor_core.append(contents[-5])
                    Ghz_value = float(contents[-3])
                    if Ghz_value > 0.0 and Ghz_value < 1.0:
                        Processor_GHz.append(1)
                    elif Ghz_value > 1.0 and Ghz_value < 2.0:
                        Processor_GHz.append(2)
                    elif Ghz_value > 2.0 and Ghz_value < 3.0:
                        Processor_GHz.append(3)

                elif "Core," in contents:  # When only name and core are present
                    Processor_name.append(contents[0])
                    Processor_core.append(contents[-3])
                    Processor_GHz.append(np.nan)
            else:
                Processor_name.append(np.nan)
                Processor_core.append(np.nan)
                Processor_GHz.append(np.nan)

        Processor_name = pd.Series(Processor_name)
        Processor_core = pd.Series(Processor_core)
        Processor_GHz = pd.Series(Processor_GHz)
        return [Processor_name, Processor_core, Processor_GHz]


class Data_processing:
    def __init__(self):
        self.paths = Data_Processing_Config()

    def initialize_data_processing(self, raw_df):
        """
        This method will take the raw dataframe as input and it will perform all the preprocessing
        steps on that dataframe by passing it through the pipeline and it will finally
        return the processed train and test dataframes after storing them in csv files.
        :param raw_df: Unprocessed raw data frame
        :return: Processed and clean train,test dataframes
        """
        try:
            fe_obj = feature_engineering()
            logging.info("Data Processing started")

            # Let's do feature engineering first
            raw_df.drop(["os"], axis=1, inplace=True)

            # Removing rupee and , character from the 'price' feature and converting to int
            for index in range(len(raw_df)):
                item_price = raw_df.loc[index, "price"]
                item_price = item_price.replace("₹", "").replace(",", "")
                item_price = re.sub(
                    r"[^\d]", "", item_price
                )  # Remove any non-digit characters
                raw_df.loc[index, "price"] = item_price

                # Let's create categories for rating feature
                item_rating = raw_df.loc[index, "rating"]
                if item_rating > 80.0:
                    raw_df.loc[index, "rating"] = "8+"
                elif item_rating > 70.0:
                    raw_df.loc[index, "rating"] = "7+"
                elif item_rating > 60.0:
                    raw_df.loc[index, "rating"] = "6+"

                item_card = raw_df.loc[index, "card"]
                if pd.isna(item_card) == False and "Memory" not in item_card.split():
                    raw_df.loc[index, "card"] = np.nan

                item_model = raw_df.loc[index, "model"]
                raw_df.loc[index, "model"] = item_model.split()[0]

            # Changing the data type of price from object to int
            raw_df["price"] = raw_df["price"].astype(np.int64)

            # Renaming the column
            raw_df.rename(columns={"model": "brand"}, inplace=True)
            raw_df.loc[raw_df["brand"] == "Oppo", "brand"] = "OPPO"
            raw_df.loc[
                raw_df["card"] == "Memory Card Supported, upto GB", "card"
            ] = "Memory Card Supported, upto 1TB"

            Sim_related_sr = fe_obj.extract_encode_sim(raw_df)
            ram_related_sr = fe_obj.extract_encode_ram(raw_df)
            battery_related_sr = fe_obj.extract_encode_battery(raw_df)
            display_related_sr = fe_obj.extract_encode_display(raw_df)
            camera_related_sr = fe_obj.extract_encode_camera(raw_df)
            processor_related_sr = fe_obj.extract_processor_info(raw_df)

            raw_df["Has_5g"] = Sim_related_sr[0]
            raw_df["Add_Features"] = Sim_related_sr[1]

            raw_df["RAM"] = ram_related_sr[0]
            raw_df["Storage"] = ram_related_sr[1]

            raw_df["Battery"] = battery_related_sr[0]
            raw_df["Charging"] = battery_related_sr[1]

            raw_df["Screen_RR"] = display_related_sr[0]
            raw_df["PPI"] = display_related_sr[1]

            raw_df["rear_cams"] = camera_related_sr[0]
            raw_df["Total_fmp"] = camera_related_sr[1]

            raw_df["Processor_name"] = processor_related_sr[0]
            raw_df["Processor_core"] = processor_related_sr[1]
            raw_df["Processor_GHz"] = processor_related_sr[2]

            # Let's fix some errors
            raw_df.loc[(raw_df["Processor_name"] == "A13"), "Processor_name"] = "Bionic"
            raw_df.loc[
                raw_df["Processor_name"] == "Sanpdragon", "Processor_name"
            ] = "Snapdragon"

            # Remvoing all the dummy phones and some outliers
            raw_df = raw_df[(raw_df["price"] > 4000) & (raw_df["price"] < 400000)]
            raw_df = raw_df[raw_df["RAM"] < 20]

            # Dropping the old features
            for feature in ["sim", "ram", "battery", "display", "camera", "processor"]:
                raw_df.drop([feature], axis=1, inplace=True)

            logging.info("Feature engineering completed")

            # Let's now build a pipeline for processing the data
            # Column transformer for univariate imputation (Mode)
            Mode_impute = ColumnTransformer(
                transformers=[
                    (
                        "Mode_imputation",
                        SimpleImputer(strategy="most_frequent"),
                        [5, 6, 7, 10, 11, 12, 13, 14],
                    )
                ],
                remainder="passthrough",
            )

            # Column transformer for the ordinal encoding
            Oridnal_enc = ColumnTransformer(
                transformers=[
                    (
                        "Oe_pcore",
                        OrdinalEncoder(
                            categories=[["Single", "Dual", "Quad", "Hexa", "Octa"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [7],
                    ),
                    (
                        "OE_rating",
                        OrdinalEncoder(
                            categories=[["6+", "7+", "8+"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [9],
                    ),
                    (
                        "Oe_card",
                        OrdinalEncoder(
                            categories=[
                                [
                                    "Memory Card Not Supported",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto 64GB",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto GB",
                                    "Memory Card Supported, upto TB",
                                    "Memory Card Supported, upto TB",
                                    "Memory Card (Hybrid)",
                                    "Memory Card (Hybrid), upto GB",
                                    "Memory Card (Hybrid), upto GB",
                                    "Memory Card (Hybrid), upto GB",
                                    "Memory Card (Hybrid), upto GB",
                                    "Memory Card (Hybrid), upto TB",
                                    "Memory Card (Hybrid), upto TB",
                                ]
                            ],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [10],
                    ),
                ],
                remainder="passthrough",
            )

            Nominal_ecn = ColumnTransformer(
                transformers=[
                    (
                        "Target_encoding",
                        ce.TargetEncoder(
                            smoothing=0.5, handle_missing="return_nan", return_df=False
                        ),
                        [9, 10],
                    ),
                ],
                remainder="passthrough",
            )

            # Column transformer for Knn imputer
            Knn_imp = ColumnTransformer(
                transformers=[
                    (
                        "Knn_imputer",
                        KNNImputer(n_neighbors=5, metric="nan_euclidean"),
                        [3, 4, 13, 14, 15],
                    )
                ],
                remainder="passthrough",
            )

            # Scaling the values
            scaling = ColumnTransformer(
                transformers=[
                    (
                        "Stand_scaling",
                        MinMaxScaler(),
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    )
                ],
                remainder="passthrough",
            )

            # Building a pipeline
            pipe = Pipeline(
                steps=[
                    ("Mode_Imputation", Mode_impute),
                    ("Ordinal_Encoding", Oridnal_enc),
                    ("Target_enc", Nominal_ecn),
                    ("KNN_Imputer", Knn_imp),
                    ("Scaling", scaling),
                ]
            )
            logging.info("Processing pipeline created")

            # Train test split
            X = raw_df.drop(["price"], axis=1)
            y = raw_df["price"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.8, random_state=1
            )
            logging.info("Train, test splitting completed")

            # Let's now process the data
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)
            logging.info("Completed processing data through pipeline")

            # Saving the pipeline
            save_object(self.paths.pipeline_path, pipe)

            return (X_train, X_test, y_train.values, y_test.values)

        except Exception as e:
            raise CustomException(e, sys)


@step
def process_data(
    path: str,
) -> Output(
    X_train=np.ndarray, X_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    try:
        # Instantiating the Data_Ingestion class
        process_data_obj = Data_processing()
        raw_df = pd.read_csv(path)
        X_train, X_test, y_train, y_test = process_data_obj.initialize_data_processing(
            raw_df
        )
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, sys)
