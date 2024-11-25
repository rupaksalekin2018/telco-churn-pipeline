import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Predicts churn based on the input features.
        Args:
            features (pd.DataFrame): Input features as a DataFrame.
        Returns:
            np.ndarray: Array of predictions.
        """
        try:
            preprocessor_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts/model_trainer", "model.pkl")

            # Load the preprocessor and model objects
            processor = load_object(preprocessor_path)
            model = load_object(model_path)

            # Transform the input features and predict
            scaled_features = processor.transform(features)
            predictions = model.predict(scaled_features)

            return predictions

        except Exception as e:
            logging.error(f"Error in PredictionPipeline.predict: {str(e)}")
            raise CustomException(e, sys)


class CustomClass:
    def __init__(self,
                 gender: str,
                 SeniorCitizen: int,
                 Partner: str,
                 Dependents: str,
                 tenure: int,
                 PhoneService: str,
                 MultipleLines: str,
                 InternetService: str,
                 OnlineSecurity: str,
                 OnlineBackup: str,
                 DeviceProtection: str,
                 TechSupport: str,
                 StreamingTV: str,
                 StreamingMovies: str,
                 Contract: str,
                 PaperlessBilling: str,
                 PaymentMethod: str,
                 MonthlyCharges: float,
                 TotalCharges: float):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def to_dataframe(self):
        """
        Converts the attributes of the custom class to a pandas DataFrame.
        Returns:
            pd.DataFrame: Input features as a DataFrame.
        """
        try:
            data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            return pd.DataFrame(data)
        except Exception as e:
            logging.error(f"Error in CustomClass.to_dataframe: {str(e)}")
            raise CustomException(e, sys)
