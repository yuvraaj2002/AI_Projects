import sys
import os
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from zenml.steps import step
from src.logger import logging
from src.exception import CustomException
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class Data_Ingestion:
    def __init__(self):
        self.raw_csv_fp = os.path.join("Artifacts", "raw_data.csv")

    def initialize_data_ingestion(self):
        """
        This method will take the path where the data is stored, and it will first store
        the raw data in the specified directory and will return us the raw_df.
        Here for simplicity we are simply using the path of csv file but here we can use any
        data source.
        :param path: Path of the raw data file
        :return: raw uncleaned and unprocessed data frame
        """
        try:
            s = Service("chromedriver.exe")
            driver = webdriver.Chrome(service=s)

            # Opening the Smartprix website's mobile section on google chrome
            driver.get("https://www.smartprix.com/")
            driver.maximize_window()

            driver.find_element(
                by=By.XPATH, value='// *[ @ id = "app"] / nav / ul / li[1] / a'
            ).click()
            time.sleep(1)

            driver.find_element(
                by=By.XPATH,
                value='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[1]/input',
            ).click()
            time.sleep(0.5)
            driver.find_element(
                by=By.XPATH,
                value='//*[@id="app"]/main/aside/div/div[5]/div[2]/label[2]/input',
            ).click()
            time.sleep(0.5)

            # Finding the old height to figure out length to scroll on website
            old_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                # Clicking the load more button

                driver.find_element(
                    by=By.XPATH, value='//*[@id="app"]/main/div[1]/div[2]/div[3]'
                ).click()

                time.sleep(0.5)

                new_height = driver.execute_script("return document.body.scrollHeight")

                print("Old height :", old_height)
                print("New height :", new_height)

                # It means we have reached to the end
                if new_height == old_height:
                    break
                old_height = new_height

            html = driver.page_source

            # Instantiating beautifulSoup
            soup = BeautifulSoup(html, "lxml")
            containers = soup.find_all(
                "div", {"class": "sm-product has-tag has-features has-actions"}
            )

            names = []
            prices = []
            ratings = []
            sim = []
            processor = []
            ram = []
            battery = []
            display = []
            camera = []
            card = []
            os = []

            for i in soup.find_all(
                "div", {"class": "sm-product has-tag has-features has-actions"}
            ):
                try:
                    names.append(i.find("h2").text)
                except:
                    names.append(np.nan)
                try:
                    prices.append(i.find("span", {"class": "price"}).text)
                except:
                    price.append(np.nan)
                try:
                    ratings.append(
                        i.find("div", {"class": "score rank-2-bg"}).find("b").text
                    )
                except:
                    ratings.append(np.nan)

                x = i.find("ul", {"class": "sm-feat specs"}).find_all("li")
                try:
                    sim.append(x[0].text)
                except:
                    sim.append(np.nan)
                try:
                    processor.append(x[1].text)
                except:
                    processor.append(np.nan)
                try:
                    ram.append(x[2].text)
                except:
                    ram.append(np.nan)
                try:
                    battery.append(x[3].text)
                except:
                    battery.append(np.nan)
                try:
                    display.append(x[4].text)
                except:
                    display.append(np.nan)
                try:
                    camera.append(x[5].text)
                except:
                    camera.append(np.nan)
                try:
                    card.append(x[6].text)
                except:
                    card.append(np.nan)
                try:
                    os.append(x[7].text)
                except:
                    os.append(np.nan)

            df = pd.DataFrame(
                {
                    "model": names,
                    "price": prices,
                    "rating": ratings,
                    "sim": sim,
                    "processor": processor,
                    "ram": ram,
                    "battery": battery,
                    "display": display,
                    "camera": camera,
                    "card": card,
                    "os": os,
                }
            )
            driver.quit()

            # Let's now store the files
            df.to_csv(self.raw_csv_fp, index=False, header=True)
            logging.info("CSV files saved successfully")

        except Exception as e:
            raise CustomException(e, sys)


@step
def ingest_data() -> None:
    try:
        # Instantiating the Data_Ingestion class
        ingest_data_obj = Data_Ingestion()
        ingest_data_obj.initialize_data_ingestion()
    except Exception as e:
        raise CustomException(e, sys)
