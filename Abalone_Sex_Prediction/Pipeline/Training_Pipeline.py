from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from Src.Ingest_data import ingest_data
from Src.Process_data import process_data
from Src.Train_model import train_model

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
    df = ingest_data("Dataset.txt")
    X_train, X_test, y_train, y_test = process_data(df)
    model = train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_pipeline()
