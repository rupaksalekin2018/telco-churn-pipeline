# Telco Customer Churn Prediction MLops Pipeline

![MLflow](https://img.shields.io/badge/MLflow-%23FF7F00.svg?style=for-the-badge&logo=mlflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-%239458FF.svg?style=for-the-badge&logo=dataversioncontrol&logoColor=white)

An end-to-end machine learning pipeline for predicting customer churn in the telecommunications industry. This project enables proactive customer retention strategies by identifying at-risk customers and key factors influencing churn.

## 📌 Overview
This repository provides a scalable solution to predict customer churn using historical data, featuring:
- **MLOps Integration**: Data versioning (DVC), experiment tracking (MLflow), and model deployment (Flask)
- **Interpretable Models**: Logistic Regression, Random Forest, and gradient-boosting algorithms optimized for recall to minimize false negatives
- **Production-Ready API**: RESTful endpoints for real-time predictions and batch inference

## 📊 Business Context
**Churn**—when customers discontinue services—costs telecom companies millions annually. By predicting churn, businesses can:
- Reduce revenue loss through targeted retention campaigns
- Improve customer satisfaction by addressing pain points identified via feature importance analysis 
- Optimize marketing budgets by focusing on high-risk customers 

**Key Stakeholders**: Marketing teams, customer support, and data science teams 

## 🗃️ Dataset
The dataset includes **7,043 customers** with 21 features, sourced from [IBM/Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn). Key features:
- **Demographics**: Gender, senior citizen status, dependents.
- **Account Details**: Tenure, contract type, payment method.
- **Service Usage**: Internet/phone subscriptions, streaming services.
- **Financials**: Monthly charges, total charges.

**Preprocessing Steps**:
1. Handle missing values in `TotalCharges` using median imputation 
2. Encode categorical variables (One-Hot Encoding) and scale numerical features (StandardScaler) 
3. Address class imbalance via SMOTE or class weighting 
## 🛠️ Technical Implementation
### Infrastructure
| Component               | Tools                                                                |
|-------------------------|----------------------------------------------------------------------|
| **Data Versioning**     | DVC                                                                  |
| **Experiment Tracking** | MLflow                                                               |
| **Model Serving**       | Flask API                                                            |                                                     |

### Model Development
- **Algorithms**: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM 
- **Hyperparameter Tuning**: GridSearchCV for optimizing recall and F1-score 
- **Evaluation Metrics**: 
  - Precision, Recall, F1-Score, ROC-AUC 

### Pipeline Architecture
```plaintext
Data Ingestion → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```
Clone the repository:
git clone https://github.com/rupaksalekin2018/telco-churn-pipeline.git
cd telco-churn-pipeline

Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

Install dependencies:
pip install -r requirements.txt

Initialize DVC and MLflow:
dvc init
mlflow UI

Training the Model
python train.py --data_path data/processed --model_type xgboost

Starting the API
python app.py  # Access endpoints at http://localhost:5000/predict

Sample Request:
```plaintext
import requests
payload = {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "Contract": "Month-to-month",
    # ... other features
}
response = requests.post("http://localhost:5000/predict", json=payload)
print(response.json())  # Output: {"churn_probability": 0.82, "prediction": "Yes"}
```
📋 Results

Best Model: LightGBM achieved 79% recall and 77% F1-score 12.

Key Drivers of Churn:

Short-term contracts (Contract_Month-to-month).

High MonthlyCharges with low tenure 311.

Lack of tech support or online security services 7.
