from zenml.steps import BaseParameters
import os

class ModelNameConfig(BaseParameters):
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False
    mode_storage_path = os.path.join("Storage", "Model.pkl")