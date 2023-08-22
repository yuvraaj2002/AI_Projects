from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from src.Data_Ingestion import ingest_data
from src.Data_Processing import process_data
from src.Model_training import train_model
from src.Model_evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(
    name="train_pipeline", enable_cache=True, settings={"docker": docker_settings}
)
def train_pipeline(step_1, step_2, step_3, step_4):
    """
    Args:
        step_1: DataClass,
        step_2: DataClass,
        step_3: DataClass,
        step_4: DataClass
    return:
        mse: float
        rmse: float
    """
    df = step_1()
    X_train, X_test, y_train, y_test = step_2("Artifacts/Raw.csv")
    model = step_3(X_train, X_test, y_train, y_test)
    mse, rmse = step_4(model, X_test, y_test)


if __name__ == "__main__":
    TP_obj = train_pipeline(
        step_1=ingest_data(),
        step_2=process_data(),
        step_3=train_model(),
        step_4=evaluation()
    )

    # Running the training pipeline
    TP_obj.run()
