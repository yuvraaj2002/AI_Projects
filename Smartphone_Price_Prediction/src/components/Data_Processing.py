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

    def find_limits(self,series):
        """
        This method will return Upper limit and Lower limit of a series
        :return:
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        IQR = q3 - q1
        Upper_lmt = q3 + 1.5 * IQR
        Lower_lmt = q1 - 1.5 * IQR
        return (Upper_lmt, Lower_lmt)


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


                # Let's cap the outliers
                Ul, Ll = self.find_limits(data['price'])

                # Capping the outliers
                data['price'] = np.where(data['price'] > Ul, Ul,
                                            np.where(data['price'] < Ll, Ll, data['price']))

            logging.info("Extracted new features from old feature and removed old features successfully")


            # Column transformer for univariate imputation (Mode)
            Mode_impute = ColumnTransformer(transformers=[
                ('Mode_imputation', SimpleImputer(strategy='most_frequent'), [8, 9, 10, 12, 13, 14, 15])
            ], remainder='passthrough')

            # Column transformer for the ordinal encoding
            Oridnal_enc = ColumnTransformer(transformers=[
                ('Oridnal_Encoding', OrdinalEncoder(categories=[
                    ['Memory Card Not Supported', 'Memory Card Supported, upto 16GB',
                     'Memory Card Supported, upto 32GB',
                     'Memory Card Supported, upto 48GB', 'Memory Card Supported, upto 64GB',
                     'Memory Card Supported, upto 128GB', 'Memory Card Supported, upto 256GB',
                     'Memory Card Supported, upto 512GB', 'Memory Card Supported, upto 1TB',
                     'Memory Card Supported, upto 2TB', 'Memory Card (Hybrid)', 'Memory Card (Hybrid), upto 64GB',
                     'Memory Card (Hybrid), upto 128GB', 'Memory Card (Hybrid), upto 256GB',
                     'Memory Card (Hybrid), upto 512GB', 'Memory Card (Hybrid), upto 1TB',
                     'Memory Card (Hybrid), upto 2TB']], handle_unknown="use_encoded_value",
                                                    unknown_value=np.nan), [10])], remainder='passthrough')

            # Column transformer for nomnial encoding
            Nom_enc = ColumnTransformer(transformers=[
                ('', ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan", return_df=False), [8, 10])
            ], remainder='passthrough')

            # Column transformer for Knn imputer
            Knn_imp = ColumnTransformer(transformers=[
                ('Knn_imputer', KNNImputer(n_neighbors=5, metric="nan_euclidean"), [2, 10, 15])
            ], remainder='passthrough')

            # Scaling the values
            scaling = ColumnTransformer(transformers=[
                ('Stand_scaling', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            ], remainder='passthrough')

            # Building a pipeline
            pipe = Pipeline(steps=[
                ('Mode_Imputation', Mode_impute),
                ('Ordinal_Encoding', Oridnal_enc),
                ('Nominal_Encoding', Nom_enc),
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

            # Let's now process the data
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