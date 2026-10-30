
import os
import joblib
import pandas as pd
# Model paths
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOC_MODEL_PATH = os.path.join(
    PACKAGE_DIR,
    "models",
    "FINAL_SOC_XGBoost_day22.joblib"
)
REMAINING_MODEL_PATH = os.path.join(
    PACKAGE_DIR,
    "models",
    "FINAL_Remaining_Charging_Time_Random_Forest_day22.joblib"
)
# Load frozen models
soc_artifact = joblib.load(SOC_MODEL_PATH)
remaining_artifact = joblib.load(REMAINING_MODEL_PATH)
soc_model = soc_artifact["model"]
remaining_model = remaining_artifact["model"]
# Read feature names directly from models
SOC_FEATURES = list(soc_model.feature_names_in_)
REMAINING_FEATURES = list(
    remaining_model.feature_names_in_
)
# SOC prediction
def predict_soc(input_data):
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    X = input_data[SOC_FEATURES]
    prediction = soc_model.predict(X)
    return prediction
# Remaining Charging Time prediction
def predict_remaining_charging_time(input_data):
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    X = input_data[REMAINING_FEATURES]
    prediction = remaining_model.predict(X)
    return prediction
# Combined prediction
def predict_battery_status(input_data):
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    soc = predict_soc(input_data)
    remaining_time = predict_remaining_charging_time(
        input_data
    )
    return {
        "SOC_prediction": float(soc[0]),
        "Remaining_Charging_Time_prediction": float(
            remaining_time[0])}
# Example
if __name__ == "__main__":
    sample = {
        "Voltage": 3.85,
        "Current": -1.50,
        "Temperature": 25.0,
        "Elapsed Time": 1800.0,
        "Cycle": 50,
        "WhAccu": 8.50,
        "Power": -5.775
    }
    result = predict_battery_status(sample)
    print("Battery Prediction")
    print("------------------")
    print("SOC:",
          result["SOC_prediction"])
    print(
        "Remaining Charging Time:",
        result[
            "Remaining_Charging_Time_prediction"
        ]
    )
