from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from Src.Ingest_data import ingest_data
from Src.Process_data import process_data
from Src.Train_model import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(
    name="train_pipeline", enable_cache=False, settings={"docker": docker_settings}
)
def train_pipeline(step_1, step_2, step_3):
    """
    Args:
        step_1: DataClass,
        step_2: DataClass,
        step_3: DataClass
    return:
        None
    """
    df = step_1("Dataset.txt")
    X_train, X_test, y_train, y_test = step_2(df)
    model = step_3(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_pipeline(
        step_1=ingest_data(),
        step_2=process_data(),
        step_3=train_model()
    ).run()
