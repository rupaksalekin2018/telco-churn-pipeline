import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object
from feast import Field, FeatureStore, Entity, FeatureView, FileSource
from feast.types import Int64, String, Float32
from feast.value_type import ValueType
from datetime import datetime, timedelta

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
    feature_store_repo_path = "feature_repo"

class DataTransformation:
    def __init__(self):
        try:
            self.data_transformation_config = DataTransformationConfig()
            
            # Get absolute path and create directory structure
            repo_path = os.path.abspath(self.data_transformation_config.feature_store_repo_path)
            os.makedirs(os.path.join(repo_path, "data"), exist_ok=True)
            
            # Create feature store yaml with minimal configuration
            feature_store_yaml_path = os.path.join(repo_path, "feature_store.yaml")
            
            # Simplified, minimal feature store configuration
            feature_store_yaml = """project: income_prediction
provider: local
registry: data/registry.db
online_store:
  type: sqlite
offline_store:
  type: file
entity_key_serialization_version: 2"""
            
            # Write configuration file
            with open(feature_store_yaml_path, 'w') as f:
                f.write(feature_store_yaml)
            
            logging.info(f"Created feature store configuration at {feature_store_yaml_path}")
            
            # Verify the configuration file content
            with open(feature_store_yaml_path, 'r') as f:
                logging.info(f"Configuration file content:\n{f.read()}")
            
            # Initialize feature store
            self.feature_store = FeatureStore(repo_path=repo_path)
            logging.info("Feature store initialized successfully")

        except Exception as e:
            logging.error(f"Error in initialization: {str(e)}")
            raise CustomException(e, sys)


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation: Setting up pipelines.")
            
            # Numerical and categorical feature lists
            numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
            categorical_features = [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService", 
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                "TechSupport", "StreamingTV", "StreamingMovies", 
                "Contract", "PaperlessBilling", "PaymentMethod"
            ]

            # Pipelines for transformation
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            # Combine transformations
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Data loaded successfully.")

            # Convert TotalCharges to numeric (handling errors)
            for df in [train_data, test_data]:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

            preprocessing_obj = self.get_data_transformation_obj()

            target_column_name = "Churn"
            input_feature_train_df = train_data.drop(columns=[target_column_name, "customerID"])
            target_feature_train_df = train_data[target_column_name].apply(lambda x: 1 if x == "Yes" else 0)

            input_feature_test_df = test_data.drop(columns=[target_column_name, "customerID"])
            target_feature_test_df = test_data[target_column_name].apply(lambda x: 1 if x == "Yes" else 0)

            logging.info("Applying preprocessing object.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved.")

            return train_arr, test_arr, self.data_transformation_config.preprocess_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def push_features_to_store(self, df, entity_id):
        try:
            # Add timestamp column if not present
            if 'event_timestamp' not in df.columns:
                df['event_timestamp'] = pd.Timestamp.now()

            # Add entity_id column if not present
            if 'entity_id' not in df.columns:
                df['entity_id'] = range(len(df))

            # Save data as parquet
            data_path = os.path.join(self.feature_store_repo_path, "data")
            parquet_path = os.path.join(data_path, f"{entity_id}_features.parquet")

            # Ensure the directory exists
            os.makedirs(data_path, exist_ok=True)

            # Save the parquet file
            df.to_parquet(parquet_path, index=False)
            logging.info(f"Saved feature data to {parquet_path}")

            # Define data source with relative path
            data_source = FileSource(
                path=f"data/{entity_id}_features.parquet",
                timestamp_field="event_timestamp"
            )

            # Define entity
            entity = Entity(
                name="entity_id",
                value_type=ValueType.INT64,
                description="Entity ID"
            )

            # Define feature view
            feature_view = FeatureView(
                name=f"{entity_id}_features",
                entities=[entity],
                schema=[
                    Field(name="gender", dtype=String),
                    Field(name="SeniorCitizen", dtype=Int64),
                    Field(name="Partner", dtype=String),
                    Field(name="Dependents", dtype=String),
                    Field(name="tenure", dtype=Int64),
                    Field(name="PhoneService", dtype=String),
                    Field(name="MultipleLines", dtype=String),
                    Field(name="InternetService", dtype=String),
                    Field(name="OnlineSecurity", dtype=String),
                    Field(name="OnlineBackup", dtype=String),
                    Field(name="DeviceProtection", dtype=String),
                    Field(name="TechSupport", dtype=String),
                    Field(name="StreamingTV", dtype=String),
                    Field(name="StreamingMovies", dtype=String),
                    Field(name="Contract", dtype=String),
                    Field(name="PaperlessBilling", dtype=String),
                    Field(name="PaymentMethod", dtype=String),
                    Field(name="MonthlyCharges", dtype=Float32),
                    Field(name="TotalCharges", dtype=Float32),
                    Field(name="Churn", dtype=String),
                ],
                source=data_source,
                online=True
            )

            # Apply to feature store
            self.feature_store.apply([entity, feature_view])
            logging.info(f"Applied entity and feature view for {entity_id}")

            # Materialize features
            self.feature_store.materialize(
                start_date=datetime.now() - timedelta(days=1),
                end_date=datetime.now() + timedelta(days=1)
            )
            logging.info("Materialized features successfully")

        except Exception as e:
            logging.error(f"Error in push_features_to_store: {str(e)}")
            raise CustomException(e, sys)

    def retrieve_features_from_store(self, entity_id):
        try:
            feature_refs = [
                f"{entity_id}_features:gender",
                f"{entity_id}_features:SeniorCitizen",
                f"{entity_id}_features:Partner",
                f"{entity_id}_features:Dependents",
                f"{entity_id}_features:tenure",
                f"{entity_id}_features:PhoneService",
                f"{entity_id}_features:MultipleLines",
                f"{entity_id}_features:InternetService",
                f"{entity_id}_features:OnlineSecurity",
                f"{entity_id}_features:OnlineBackup",
                f"{entity_id}_features:DeviceProtection",
                f"{entity_id}_features:TechSupport",
                f"{entity_id}_features:StreamingTV",
                f"{entity_id}_features:StreamingMovies",
                f"{entity_id}_features:Contract",
                f"{entity_id}_features:PaperlessBilling",
                f"{entity_id}_features:PaymentMethod",
                f"{entity_id}_features:MonthlyCharges",
                f"{entity_id}_features:TotalCharges",
                f"{entity_id}_features:Churn",
            ]

            feature_vector = self.feature_store.get_online_features(
                feature_refs=feature_refs,
                entity_rows=[{"entity_id": i} for i in range(len(df))]
            ).to_df()

            logging.info(f"Retrieved features for {entity_id}")
            return feature_vector

        except Exception as e:
            logging.error(f"Error in retrieve_features_from_store: {str(e)}")
            raise CustomException(e, sys)