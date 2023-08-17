from zenml.steps import BaseParameters
import os


class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "Logistic Regression"
    fine_tuning: bool = False
    model_storage_path = os.path.join("Artifacts", "Model.pkl")
