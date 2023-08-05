from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
import numpy as np
import re
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.Data_Processing import Data_processing
from src.components.Model_training import Train_Model


@dataclass
class Data_Ingestion_Config:
    raw_structured_data_path = os.path.join("artifacts", "raw_structured.csv")


class Data_Ingestion:

    def __init__(self):
        self.data_paths = Data_Ingestion_Config()


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

    def initialize_data_ingestion(self, raw_data_path):
        try:
            logging.info("Data Ingestion initialized")

            # Let's first read the raw data from source
            df = pd.read_csv(raw_data_path)
            logging.info("Reading csv file completed successfully")

            # Let's first remove some unnecessary columns
            df.drop(['os'], axis=1, inplace=True)
            logging.info("Removed unnecessary column")

            # Removing rupee and , character from the 'price' feature and converting to int
            for index in range(len(df)):
                item_price = df.loc[index, 'price']
                item_price = item_price.replace("₹", "").replace(",", "")
                item_price = re.sub(r'[^\d]', '', item_price)  # Remove any non-digit characters
                df.loc[index, 'price'] = item_price

                # Let's create categories for rating feature
                item_rating = df.loc[index, 'rating']
                if item_rating > 80.0:
                    df.loc[index, 'rating'] = '8+'
                elif item_rating > 70.0:
                    df.loc[index, 'rating'] = '7+'
                elif item_rating > 60.0:
                    df.loc[index, 'rating'] = '6+'

            df['price'] = df['price'].astype(np.int64)
            logging.info("Cleaned the price feature and created categories for rating feature")

            list_series_sim = self.extract_encode_sim(data=df)
            list_series_ram = self.extract_encode_ram(data=df)
            list_series_battery = self.extract_encode_battery(data=df)
            list_seires_display = self.extract_encode_display(data=df)
            list_series_camera = self.extract_encode_camera(data=df)
            list_processor_info = self.extract_processor_info(data=df)

            df['Has_5g'] = list_series_sim[0]
            df['Add_Features'] = list_series_sim[1]

            df['RAM'] = list_series_ram[0]
            df['Storage'] = list_series_ram[1]

            df['Battery'] = list_series_battery[0]
            df['Charging'] = list_series_battery[1]

            df['Screen_RR'] = list_seires_display[0]
            df['PPI'] = list_seires_display[1]

            df['rear_cams'] = list_series_camera[0]
            df['Total_fmp'] = list_series_camera[1]

            df['Processor_name'] = list_processor_info[0]
            df['Processor_core'] = list_processor_info[1]
            df['Processor_GHz'] = list_processor_info[2]
            # Let's fix some errors in the processor_name feature
            df.loc[(df['Processor_name'] == "A13"), 'Processor_name'] = "Bionic"
            df.loc[df['Processor_name'] == 'Sanpdragon', 'Processor_name'] = "Snapdragon"

            # Let's now drop all the old features
            for feature in ['sim', 'ram', 'display', 'camera', 'battery', 'processor']:
                df.drop([feature], axis=1, inplace=True)

            # Removing any unrelated information from the card feature and extracting the name of smartphone brand
            for i in range(len(df)):
                item_model = df.loc[i, 'model']
                df.loc[i, 'model'] = item_model.split()[0]

                item_card = df.loc[i, 'card']
                if pd.isna(item_card) == False and 'Memory' not in item_card.split():
                    df.loc[i, 'card'] = np.nan

            # Renaming the column and fixing some incorrect values in both model and card feature
            df.rename(columns={'model': 'brand'}, inplace=True)
            df.loc[df['brand'] == 'Oppo', 'brand'] = 'OPPO'
            df.loc[df["card"] == "Memory Card Supported, upto 1000GB", "card"] = "Memory Card Supported, upto 1TB"

            # Removing dummy phones and outliers from the features
            data = df[(df['price'] > 4000) & (df['price'] < 400000)]
            data = data[data['RAM'] < 20]
            df.drop(['Battery', 'Processor_GHz'], axis=1, inplace=True)

            logging.info("Extracted new features from old feature and removed old features successfully")


            # Let's make the directories to store the files
            os.makedirs(os.path.dirname(self.data_paths.raw_structured_data_path), exist_ok=True)

            # Let's now store the files
            df.to_csv(self.data_paths.raw_structured_data_path, index=False, header=True)
            logging.info("Raw structured file saved")

            logging.info("Data Ingestion completed")
            return (self.data_paths.raw_structured_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    data_ingestion_obj = Data_Ingestion()
    raw_data_path = data_ingestion_obj.initialize_data_ingestion("Data.csv")

    data_processing_obj = Data_processing()
    X_train, y_train, X_test, y_test,pipe_path = data_processing_obj.initialize_data_processing(raw_data_path)

    model_training_obj = Train_Model()
    Avg_R2 = model_training_obj.initialize_model_training(raw_data_path,pipe_path,X_train, y_train, X_test, y_test)
    print("Average R2 Score : ",Avg_R2)
