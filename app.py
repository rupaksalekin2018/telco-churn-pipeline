from flask import Flask, request, jsonify
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomClass

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        json_data = request.get_json()

        # Create CustomClass instance with JSON data
        data = CustomClass(
            gender=json_data.get("gender"),
            SeniorCitizen=int(json_data.get("SeniorCitizen")),
            Partner=json_data.get("Partner"),
            Dependents=json_data.get("Dependents"),
            tenure=int(json_data.get("tenure")),
            PhoneService=json_data.get("PhoneService"),
            MultipleLines=json_data.get("MultipleLines"),
            InternetService=json_data.get("InternetService"),
            OnlineSecurity=json_data.get("OnlineSecurity"),
            OnlineBackup=json_data.get("OnlineBackup"),
            DeviceProtection=json_data.get("DeviceProtection"),
            TechSupport=json_data.get("TechSupport"),
            StreamingTV=json_data.get("StreamingTV"),
            StreamingMovies=json_data.get("StreamingMovies"),
            Contract=json_data.get("Contract"),
            PaperlessBilling=json_data.get("PaperlessBilling"),
            PaymentMethod=json_data.get("PaymentMethod"),
            MonthlyCharges=float(json_data.get("MonthlyCharges")),
            TotalCharges=float(json_data.get("TotalCharges")),
        )

        # Convert input data to DataFrame
        input_features = data.to_dataframe()

        # Run prediction
        pipeline = PredictionPipeline()
        predictions = pipeline.predict(input_features)

        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port='5001')


'''curl -X POST http://127.0.0.1:5001/predict \
-H "Content-Type: application/json" \
-d '{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 840.50
}'
'''