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
        Has_5g = []
        Add_Features = []

        for item in data['sim']:
            Has_5g.append(1 if '5G,' in item.split() else 0)
            Add_Features.append(1 if 'NFC,' in item.split() or 'IR,' in item.split(',') else 0)

        Has_5g = pd.Series(Has_5g)
        Add_Features = pd.Series(Add_Features)
        return [Has_5g,Add_Features]

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

                if ('Battery' in contents) and ('Charging' in contents) and len(contents) == 7:  # Means the W is given
                    Battery.append(float(contents[0]))
                    Charging.append(float(contents[4][:-1]))

                # Means keywords are present but value of fast charging is not mentioned
                elif ('Battery' in contents) and ('Charging' in contents) and len(contents) == 6:
                    Battery.append(float(contents[0]))
                    Charging.append(np.nan)

                elif 'Battery' in contents:
                    Battery.append(float(contents[0]))
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
        PPI = []
        Screen_RR = []

        for item in data['display']:
            contents = item.split()
            if ('inches,' in contents) and ('Display'):  # Filtered out any wrong info

                diagonal_len = float(contents[0])
                px1_val = float(contents[2])
                px2_val = float(contents[4])
                total_px_value = px1_val * px2_val
                PPI.append(int(total_px_value / diagonal_len))
            else:
                PPI.append(np.nan)

        for item in data['display']:
            contents = item.split()
            if 'Hz' in contents:
                Screen_RR.append(int(contents[6]))
            else:
                Screen_RR.append(np.nan)

        PPI = pd.Series(PPI)
        Screen_RR = pd.Series(Screen_RR)
        return [Screen_RR,PPI]


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


    def extract_processor_info(self,data):

        Processor_name = []
        Processor_core = []
        Processor_GHz = []

        for item in data['processor']:
            contents = item.split()
            if 'Processor' in contents:  # Filtered out wrong info

                if ('Core,' in contents) and ('GHz' in contents):  # When all name,core and Ghz are present
                    Processor_name.append(contents[0])
                    Processor_core.append(contents[-5])
                    Processor_GHz.append(float(contents[-3]))

                elif 'Core,' in contents:  # When only name and core are present
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
        return [Processor_name,Processor_core,Processor_GHz]


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
            logging.info("Removed unnecessary column")

            # Removing rupee and , character from the 'price' feature and converting to int
            for data in [train_data,test_data]:
                for index in range(len(data)):
                    item_price = data.loc[index, 'price']
                    item_price = item_price.replace("₹", "").replace(",", "")
                    item_price = re.sub(r'[^\d]', '', item_price)  # Remove any non-digit characters
                    data.loc[index, 'price'] = item_price

                    # Let's create categories for rating
                    item_rating = data.loc[index, 'rating']
                    if item_rating > 80.0:
                        data.loc[index, 'rating'] = '8+'
                    elif item_rating > 70.0:
                        data.loc[index, 'rating'] = '7+'
                    elif item_rating > 60.0:
                        data.loc[index, 'rating'] = '6+'

                data['price'] = data['price'].astype(np.int64)
            logging.info("Cleaned the price feature and created categories for rating feature")


            #Extracting features from sim feature of training and testing data
            for data in [train_data,test_data]:
                list_series_sim = self.extract_encode_sim(data=data)
                list_series_ram = self.extract_encode_ram(data=data)
                list_series_battery = self.extract_encode_battery(data=data)
                list_seires_display = self.extract_encode_display(data=data)
                list_series_camera = self.extract_encode_camera(data=data)
                list_processor_info = self.extract_processor_info(data = data)

                data['Has_5g'] = list_series_sim[0]
                data['Add_Features'] = list_series_sim[1]

                data['RAM'] = list_series_ram[0]
                data['Storage'] = list_series_ram[1]

                data['Battery'] = list_series_battery[0]
                data['Charging'] = list_series_battery[1]

                data['Screen_RR'] = list_seires_display[0]
                data['PPI'] = list_seires_display[1]

                data['rear_cams'] = list_series_camera[0]
                data['Total_fmp'] = list_series_camera[1]

                data['Processor_name'] = list_processor_info[0]
                data['Processor_core'] = list_processor_info[1]
                data['Processor_GHz'] = list_processor_info[2]
                # Let's fix some errors in the processor_name feature
                data.loc[(data['Processor_name'] == "A13"), 'Processor_name'] = "Bionic"
                data.loc[data['Processor_name'] == 'Sanpdragon', 'Processor_name'] = "Snapdragon"

                # Let's now drop all the old features
                for feature in ['sim','ram','display','camera','battery','processor']:
                    data.drop([feature],axis =1,inplace=True)

                # Remvoing any unrelated information from the card feature and extracting the name of smartphone brand
                for i in range(len(data)):
                    item_model = data.loc[i, 'model']
                    data.loc[i, 'model'] = item_model.split()[0]

                    item_card = data.loc[i, 'card']
                    if pd.isna(item_card) == False and 'Memory' not in item_card.split():
                        data.loc[i, 'card'] = np.nan

                # Renaming the column and fixing some incorrect values in both model and card feature
                data.rename(columns={'model': 'brand'}, inplace=True)
                data.loc[data['brand'] == 'Oppo', 'brand'] = 'OPPO'
                data.loc[data["card"] == "Memory Card Supported, upto 1000GB", "card"] = "Memory Card Supported, upto 1TB"

                # Removing dummy phones and outliers from the features
                data = data[(data['price'] > 4000) & (data['price'] < 400000)]
                data = data[data['RAM'] < 20]

            logging.info("Extracted new features from old feature and removed old features successfully")

            # Column transformer for univariate imputation (Mode)
            # Column transformer for univariate imputation (Mode)
            Mode_impute = ColumnTransformer(transformers=[
                ('Mode_imputation', SimpleImputer(strategy='most_frequent'), [5,6,9, 10, 11, 12, 13])
            ], remainder='passthrough')

            # Column transformer for the ordinal encoding
            Oridnal_enc = ColumnTransformer(transformers=[
                ('OE_pcore', OrdinalEncoder(categories=[['Single', 'Dual', 'Quad', 'Hexa', 'Octa']],
                                            handle_unknown="use_encoded_value", unknown_value=np.nan), [6]),
                ('OE_rating', OrdinalEncoder(categories=[['6+', '7+', '8+']], handle_unknown="use_encoded_value",
                                             unknown_value=np.nan), [8]),
                ('OE_card', OrdinalEncoder(categories=[
                    ['Memory Card Not Supported', 'Memory Card Supported, upto 16GB',
                     'Memory Card Supported, upto 32GB',
                     'Memory Card Supported, upto 48GB', 'Memory Card Supported, upto 64GB',
                     'Memory Card Supported, upto 128GB', 'Memory Card Supported, upto 256GB',
                     'Memory Card Supported, upto 512GB', 'Memory Card Supported, upto 1TB',
                     'Memory Card Supported, upto 2TB', 'Memory Card (Hybrid)', 'Memory Card (Hybrid), upto 64GB',
                     'Memory Card (Hybrid), upto 128GB', 'Memory Card (Hybrid), upto 256GB',
                     'Memory Card (Hybrid), upto 512GB', 'Memory Card (Hybrid), upto 1TB',
                     'Memory Card (Hybrid), upto 2TB']], handle_unknown="use_encoded_value",
                                           unknown_value=np.nan), [9])], remainder='passthrough')

            # Column transformer for nomnial encoding
            Nom_enc = ColumnTransformer(transformers=[
                ('', ce.TargetEncoder(smoothing=0.2, handle_missing="return_nan", return_df=False), [8,9])
            ], remainder='passthrough')

            # Column transformer for Knn imputer
            Knn_imp = ColumnTransformer(transformers=[
                ('Knn_imputer', KNNImputer(n_neighbors=5, metric="nan_euclidean"), [3, 4, 12, 13])
            ], remainder='passthrough')

            # Scaling the values
            scaling = ColumnTransformer(transformers=[
                ('Stand_scaling', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
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
            X_train = train_data.drop(['price','Battery','Processor_GHz'], axis=1)
            y_train = train_data['price']

            X_test = test_data.drop(['price','Battery','Processor_GHz'], axis=1)
            y_test = test_data['price']
            logging.info("Created X_train,y_train and X_test,y_test successfully")


            #Let's now process the data
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