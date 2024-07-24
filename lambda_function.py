import json
import boto3
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.exceptions import NotFittedError

local_model_key = f"model.pkl"
def lambda_handler(event, context):
    # CORS handling
    origin = event.get("headers", {}).get("origin", "*")

    # Validating and extracting query parameters
    query_params = event.get("queryStringParameters", {})
    required_params = ["age", "sex", "bmi", "children", "smoker", "region"]

    if not all(param in query_params for param in required_params):
        return {"statusCode": 400, "body": json.dumps({"error": "Missing required query parameters"})}

    try:
        age = int(query_params["age"])
        sex = query_params["sex"]
        bmi = float(query_params["bmi"])
        children = int(query_params["children"])
        smoker = query_params["smoker"]
        region = query_params["region"]
    except ValueError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid parameter types"})}

    # Load model
    if not os.path.exists(local_model_key):
        boto3.client("s3").download_file(bucket_name, remote_model_key, local_model_key)

    # Predict
    try:
        trained_model = joblib.load(local_model_key)
        input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns=["age", "sex", "bmi", "children", "smoker", "region"])
        prediction = trained_model.predict(input_data)
        result = int(np.round(prediction[0]))
    except NotFittedError:
        return {"statusCode": 500, "body": json.dumps({"error": "Model not fitted"})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": f"Prediction error: {str(e)}"})}

    # Response
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": origin
        },
        "body": json.dumps({"result": result})
    }
