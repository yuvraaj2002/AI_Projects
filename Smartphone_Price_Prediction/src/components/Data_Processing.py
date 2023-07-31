from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
import re
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


class Data_processing:
    def __init__(self):
        self.pipe_path = Data_Processing_Config()


    def extract_encode_sim(self,data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        # Empty lists to store data
        Sim_type = []
        Has_5g = []
        Add_Features = []
        Volte_Wifi = []

        for item in data['sim']:
            Sim_type.append(1 if 'Dual Sim' in item.split(',') else 0)
            Has_5g.append(2 if ' 5G' in item.split(',') else (1 if ' 4G' in item.split(',') or ' 3G' in item.split(',') else 0))
            Add_Features.append(1 if ' NFC' in item.split(',') or ' IR Blaster' in item.split(',') else 0)
            Volte_Wifi.append(1 if ' VoLTE' in item.split(',') and ' Wi-Fi' in item.split(',') else 0)

        Sim_type = pd.Series(Sim_type)
        Has_5g = pd.Series(Has_5g)
        Add_Features = pd.Series(Add_Features)
        Volte_Wifi = pd.Series(Volte_Wifi)
        return [Sim_type,Has_5g,Add_Features,Volte_Wifi]

    def extract_encode_ram(self,data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        RAM = []
        Storage = []

        for item in data['ram']:
            contents = item.split()
            if ('RAM,' in contents) or ('inbuilt' in contents):  # Filtered out any wrong info

                # Now we just need to figure out if both are present or only one is present
                if ('RAM,' in contents) and ('inbuilt' in contents):
                    RAM.append(int(contents[0]))
                    Storage.append(int(contents[3]))
                elif 'RAM,' in contents:
                    RAM.append(int(contents[0]))
                    Storage.append(np.nan)
                elif 'inbuilt' in contents:
                    RAM.append(np.nan)
                    Storage.append(int(contents[0]))

            else:
                RAM.append(np.nan)
                Storage.append(np.nan)

        RAM = pd.Series(RAM)
        Storage = pd.Series(Storage)
        return [RAM,Storage]


    def extract_encode_battery(self,data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        Battery = []
        Charging = []

        for item in data['battery']:
            contents = item.split()
            if ('Battery' in contents) or ('Charging' in contents):  # Filtered out any wrong info

                if ('Battery' in contents) and ('Charging' in contents) and len(contents) == 7:
                    Battery.append(int(contents[0]))
                    Charging.append(float(contents[4][:-1]))

                # Means keywords are present but value of fast charging is not mentioned
                elif ('Battery' in contents) and ('Charging' in contents) and len(contents) == 6:
                    Battery.append(int(contents[0]))
                    Charging.append(np.nan)

                elif 'Battery' in contents:
                    Battery.append(int(contents[0]))
                    Charging.append(np.nan)

            else:
                Battery.append(np.nan)
                Charging.append(np.nan)

        Battery = pd.Series(Battery)
        Charging = pd.Series(Charging)
        return [Battery,Charging]


    def extract_encode_display(self,data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        diagonal_len = []
        Total_px = []

        for item in data['display']:
            contents = item.split()
            if ('inches,' in contents) and ('Display'):  # Filtered out any wrong info

                diagonal_len.append(float(contents[0]))
                px1_val = contents[2]
                px2_val = contents[4]
                total_px_value = int(px1_val) * int(px2_val)
                Total_px.append(total_px_value)

            else:
                diagonal_len.append(np.nan)
                Total_px.append(np.nan)

        diagonal_len = pd.Series(diagonal_len)
        Total_px = pd.Series(Total_px)
        return [diagonal_len,Total_px]


    def extract_encode_camera(self,data):
        """
        This method will take data and will return list of pandas series of
        newly created features with encoded values from the sim feature of that data
        :param data
        :return: list of pandas series
        """
        rear_cams = []
        Total_fmp = []

        data['camera'].fillna(data['camera'].mode()[0], inplace=True)

        for item in data['camera']:
            contents = item.split()
            if ('Rear' in contents) and ('Front' in contents) and ('&' in contents):  # Filtered out any wrong info

                if 'Quad' in contents:
                    rear_cams.append(4)
                elif 'Triple' in contents:
                    rear_cams.append(3)
                elif 'Dual' in contents:
                    rear_cams.append(2)
                else:
                    rear_cams.append(1)

                Total_fmp.append(float(item.split('&')[-1].split()[0]))

            else:
                rear_cams.append(np.nan)
                Total_fmp.append(np.nan)

        rear_cams = pd.Series(rear_cams)
        Total_fmp = pd.Series(Total_fmp)
        return [rear_cams,Total_fmp]


    def initialize_data_processing(self, train_data_path, test_data_path):
        try:
            logging.info("Data Processing started")

            # Let's load the training and testing data
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info("Training and testing data loaded")

            # Let's first remove some unnecessary columns
            train_data.drop(['os'],axis=1,inplace=True)
            test_data.drop(['os'],axis=1,inplace=True)
            logging.info("Removed unnecessary columns")

            # Removing rupee and , character from the 'price' feature and converting to int
            for data in [train_data,test_data]:
                for index in range(len(data)):
                    item = data.loc[index, 'price']
                    item = item.replace("₹", "").replace(",", "")
                    item = re.sub(r'[^\d]', '', item)  # Remove any non-digit characters
                    data.loc[index, 'price'] = item

                data['price'] = data['price'].astype(np.int64)
            logging.info("Cleaned the price feature")


            #Extracting features from sim feature of training and testing data
            for data in [train_data,test_data]:
                list_series_sim = self.extract_encode_sim(data=data)
                list_series_ram = self.extract_encode_ram(data=data)
                list_series_battery = self.extract_encode_battery(data=data)
                list_seires_display = self.extract_encode_display(data=data)
                list_series_camera = self.extract_encode_camera(data=data)

                data['Sim_type'] = list_series_sim[0]
                data['Has_5g'] = list_series_sim[1]
                data['Add_Features'] = list_series_sim[2]
                data['Volte_Wifi'] = list_series_sim[3]

                data['RAM'] = list_series_ram[0]
                data['Storage'] = list_series_ram[1]

                data['Battery'] = list_series_battery[0]
                data['Charging'] = list_series_battery[1]

                data['diagonal_len'] = list_seires_display[0]
                data['Total_px'] = list_seires_display[1]

                data['rear_cams'] = list_series_camera[0]
                data['Total_fmp'] = list_series_camera[1]

                # Let's now drop all the old features
                for feature in ['sim','ram','battery','display','camera']:
                    data.drop([feature],axis =1,inplace=True)

                # Remvoing any unrelated information for card feature
                for i in range(len(data)):
                    item = data.loc[i, 'card']
                    if pd.isna(item) == False and 'Memory' not in item.split():
                        data.loc[i, 'card'] = np.nan

                data.loc[data["card"] == "Memory Card Supported, upto 1000GB", "card"] = "Memory Card Supported, upto 1TB"

                # Considering only the name of the processor and model
                for i in range(len(data)):
                    item1 = data.loc[i, 'processor']
                    item2 = data.loc[i, 'model']
                    data.loc[i, 'processor'] = item1.split()[0]
                    data.loc[i, 'model'] = item2.split()[0]

                # Let's fix some errors in the processor feature
                data.loc[(data["processor"] == "A13") | (data["processor"] == "Apple"), "processor"] = "Bionic"
                data.loc[data["processor"] == "Helio,", "processor"] = "Helio"
                data.loc[data["processor"] == "Unisoc,", "processor"] = "Unisoc"
                data.loc[data["processor"] == "Sanpdragon", "processor"] = "Snapdragon"

                # Sort of doing mode imputation of irrelevent information
                types = ['1', '4', 'SC9863A,', '32', '1.77', 'SC6531E,', '48', '256', 'Single', 'Samsung', '(28',
                         'Fusion','52', '2000', '800', '1450', 'Dual', '8']

                for type in types:
                    data.loc[data["processor"] == type, "processor"] = "Snapdragon"

            logging.info("Extracted new features from old feature and removed old features successfully")
            print(train_data.head(4))



            #
            # # Let's remove any duplicates
            # train_data.drop_duplicates(inplace=True)
            # test_data.drop_duplicates(inplace=True)
            # logging.info("Removed duplicates")
            #
            # # Let's make X_train,y_train and X_test,y_test
            # X_train = train_data.drop(["target"], axis=1)
            # y_train = train_data["target"]
            # X_test = test_data.drop(["target"], axis=1)
            # y_test = test_data["target"]
            # logging.info("Created X_train,y_train and X_test,y_test")
            #
            # # Let's now build a pipeline
            # # Define the column transformer for imputation
            # Simple_impute_transformer = ColumnTransformer(
            #     transformers=[
            #         ("mean_imputer", SimpleImputer(strategy="mean"), [0, 6, 9]),
            #         ("mode_imputer", SimpleImputer(strategy="most_frequent"), [3, 4]),
            #     ],
            #     remainder="passthrough",
            # )
            #
            # # Define the column transformer for encoding
            # encode_values = ColumnTransformer(
            #     transformers=[
            #         (
            #             "Encode_ordinal_Re",
            #             OrdinalEncoder(
            #                 categories=[
            #                     ["No relevent experience", "Has relevent experience"]
            #                 ],
            #                 handle_unknown="use_encoded_value",
            #                 unknown_value=np.nan,
            #             ),
            #             [6],
            #         ),
            #         (
            #             "Encode_ordinal_eu",
            #             OrdinalEncoder(
            #                 categories=[
            #                     [
            #                         "no_enrollment",
            #                         "Part time course",
            #                         "Full time course",
            #                     ]
            #                 ],
            #                 handle_unknown="use_encoded_value",
            #                 unknown_value=np.nan,
            #             ),
            #             [3],
            #         ),
            #         (
            #             "Encode_ordinal_el",
            #             OrdinalEncoder(
            #                 categories=[
            #                     [
            #                         "Primary School",
            #                         "High School",
            #                         "Graduate",
            #                         "Masters",
            #                         "Phd",
            #                     ]
            #                 ],
            #                 handle_unknown="use_encoded_value",
            #                 unknown_value=np.nan,
            #             ),
            #             [4],
            #         ),
            #         (
            #             "Encode_ordinal_cs",
            #             OrdinalEncoder(
            #                 categories=[
            #                     [
            #                         "<10",
            #                         "10/49",
            #                         "50-99",
            #                         "100-500",
            #                         "500-999",
            #                         "1000-4999",
            #                         "5000-9999",
            #                         "10000+",
            #                     ]
            #                 ],
            #                 handle_unknown="use_encoded_value",
            #                 unknown_value=np.nan,
            #             ),
            #             [8],
            #         ),
            #         (
            #             "Encode_target_major",
            #             ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
            #             [7],
            #         ),
            #         (
            #             "Encode_target_ct",
            #             ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
            #             [9],
            #         ),
            #         (
            #             "Encode_target_gen",
            #             ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan"),
            #             [5],
            #         ),
            #     ],
            #     remainder="passthrough",
            # )
            #
            # Knn_imputer = ColumnTransformer(
            #     transformers=[
            #         (
            #             "Knn_Imputer",
            #             KNNImputer(n_neighbors=5, metric="nan_euclidean"),
            #             [3, 4, 5, 6],
            #         )
            #     ],
            #     remainder="passthrough",
            # )
            #
            # yeo_transformation = ColumnTransformer(
            #     transformers=[("Yeo-Johnson", PowerTransformer(), [9])],
            #     remainder="passthrough",
            # )
            #
            # # Column transformer to do feature scaling
            # scaling_transformer = ColumnTransformer(
            #     transformers=[("scale_transformer", MinMaxScaler(), [1])],
            #     remainder="passthrough",
            # )
            #
            # # Define the final pipeline
            # pipe = Pipeline(
            #     steps=[
            #         ("impute_transformer", Simple_impute_transformer),
            #         ("encode_values", encode_values),
            #         ("Knn_imputer", Knn_imputer),
            #         ("Yeo-Johnson-Transformation", yeo_transformation),
            #         ("Scaling", scaling_transformer),
            #     ]
            # )
            #
            # # Let's train and test data
            # X_train = pipe.fit_transform(X_train, y_train)
            # X_test = pipe.transform(X_test)
            # logging.info("Train and test data processed")
            #
            # y_train = np.array(y_train.values)
            # y_test = np.array(y_test.values)
            #
            # # Let's now save the pipeline
            # save_object(file_path=self.pipe_path.pipeline_path, obj=pipe)
            # logging.info("Saved pipeilne object")
            #
            # logging.info("Data Processing completed")
            # return (X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)
