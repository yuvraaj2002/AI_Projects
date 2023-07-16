import sys
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def Make_prediction(self, model_features):
        try:
            # Let's load the trained model
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            # Let's load the preprocessing pipeline
            preprocessor_path = os.path.join("artifacts", "Processing_pipe.pkl")
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Model and processing pipeline loaded")

            # Let's now process the data
            model_features = preprocessor.transform(model_features)
            prediction = model.predict(model_features)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)
