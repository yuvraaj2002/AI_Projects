from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from Src.Data_Ingestion import ingest_data
from Src.Data_Processing import process_data_step
from Src.Model_Training import train_model
from Src.Evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(
    name="train_pipeline", enable_cache=False, settings={"docker": docker_settings}
)
def train_pipeline():
    """
    Args:
        step_1: DataClass,
        step_2: DataClass,
        step_3: DataClass
    return:
        None
    """
    Train_df, Test_df = ingest_data(
        "/home/yuvraj/Documents/AI/AI_Projects/Find_Home.AI/Notebook_And_Dataset/Cleaned_datasets/Combined_Cleandata_V4.csv"
    )
    X_train, X_test, y_train, y_test = process_data_step(
        Train_df=Train_df, Test_df=Test_df
    )

    regressor = train_model(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )
    R2_score, MAE = evaluation(
        model=regressor, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
    )


if __name__ == "__main__":
    train_pipeline()
