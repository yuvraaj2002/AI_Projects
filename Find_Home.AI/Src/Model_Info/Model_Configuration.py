from zenml.steps import BaseParameters
import os


class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "Xgboost"
    fine_tuning: bool = True
    model_storage_path = os.path.join("Artifacts", "Model.pkl")
