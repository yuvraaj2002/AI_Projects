from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from Src.Data_Ingestion import ingest_data
from Src.Data_Processing import process_data_step
docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(name="train_pipeline", enable_cache=True, settings={"docker": docker_settings})
def train_pipeline():
    """
    Args:
        step_1: DataClass,
        step_2: DataClass
    return:
        None
    """
    Train_df,Test_df = ingest_data("/home/yuvraj/Github/Machine_Learning_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Combined_CleanData_V4.csv")
    X_train, X_test, y_train, y_test = process_data_step(Train_df=Train_df,Test_df=Test_df)


if __name__ == "__main__":
    train_pipeline()