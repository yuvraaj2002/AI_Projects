import sys
import pandas as pd
import os
import pickle
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def Make_prediction(self, model_features):
        try:
            # Let's load the trained model
            model_path = os.path.join("artifacts", "Model.pkl")
            preprocessor_path = os.path.join("artifacts", "Processing_pipeline.pkl")
            model = load_object(model_path)
            pipeline = load_object(preprocessor_path)

            # Let's now process the data
            model_features = pipeline.transform(model_features)
            prediction = model.predict(model_features)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)
