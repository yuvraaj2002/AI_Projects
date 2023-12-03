import os
from dotenv import load_dotenv
from zenml.steps import step
import mysql.connector
import pandas as pd

# For loading environment variables from .env file
load_dotenv()
mysql_password = os.getenv("DB_password")

class Data_Ingestion:
    def __init__(self):
        self.raw_data_path = os.path.join("Data", "raw_data.csv")


    def initialize_data_ingestion(self):

        mysql_db = mysql.connector.connect(
            host = '',
            user = 'root',
            password = mysql_password,
            port = '3306',
            database = 'RP'
        )

        # MYSQL Cursor object is used to execute statement and communicate with database
        mycursor = mysql_db.cursor()
        mycursor.execute('SELECT * FROM RP.Products;')

        # Let's fetch the data (We will get list of tuples in return)
        data = mycursor.fetchall()

        # Creating a pandas dataframe
        df = pd.DataFrame(data,columns = ['product_id', 'product_category_name', 'month_year', 'qty',
       'total_price', 'freight_price', 'unit_price', 'product_name_lenght',
       'product_description_lenght', 'product_photos_qty', 'product_weight_g',
       'product_score', 'customers', 'weekday', 'weekend', 'holiday', 'month',
       'year', 's', 'volume', 'comp_1', 'ps1', 'fp1', 'comp_2', 'ps2', 'fp2',
       'comp_3', 'ps3', 'fp3', 'lag_price'])

        # Let's now store the files
        df.to_csv(self.raw_data_path, index=False, header=True)


@step
def Ingest_data() -> None:
    try:
        # Instantiating the Data_Ingestion class
        ingest_data_obj = Data_Ingestion()
        ingest_data_obj.initialize_data_ingestion()
    except Exception as e:
        raise e
