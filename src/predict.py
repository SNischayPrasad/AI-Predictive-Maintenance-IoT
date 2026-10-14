"""
predict.py
----------
Uses the saved model to predict failure for NEW sensor readings.
This is what would run on an edge device / cloud service in production.
"""

import pandas as pd

from src import config
from src.alert_system import format_alert, generate_alert
from src.feature_engineering import add_features
from src.model_training import load_model


def build_reading(product_type, air_temp, process_temp, rpm, torque, tool_wear):
    """Create a one-row DataFrame in the same raw format the sensors send."""
    return pd.DataFrame([{
        config.COL_TYPE: product_type,
        config.COL_AIR_TEMP: air_temp,
        config.COL_PROCESS_TEMP: process_temp,
        config.COL_RPM: rpm,
        config.COL_TORQUE: torque,
        config.COL_TOOL_WEAR: tool_wear,
    }])


def predict_failure(raw_df, bundle=None):
    """Return a DataFrame with failure probability + prediction for each raw reading."""
    bundle = bundle or load_model()
    features = add_features(raw_df)[bundle["features"]]
    prob = bundle["pipeline"].predict_proba(features)[:, 1]
    result = features.copy()
    result["failure_probability"] = prob
    result["prediction"] = (prob >= 0.5).astype(int)
    result["prediction_label"] = result["prediction"].map({0: "No Failure", 1: "FAILURE"})
    return result


def predict_single(machine_id="MACHINE-01", **sensor_values):
    """Predict for one reading and print an alert. Returns the alert dict."""
    raw = build_reading(**sensor_values)
    result = predict_failure(raw)
    row = result.iloc[0]
    alert = generate_alert(machine_id, row.to_dict(), row["failure_probability"])
    print(format_alert(alert))
    return alert


def demo_predictions():
    """Three hand-made scenarios that show the model reacting to machine health."""
    scenarios = [
        ("MACHINE-01 (healthy)",
         dict(product_type="M", air_temp=298.5, process_temp=309.0, rpm=1500, torque=40.0, tool_wear=30)),
        ("MACHINE-02 (worn tool, high load)",
         dict(product_type="L", air_temp=300.0, process_temp=310.5, rpm=1380, torque=58.0, tool_wear=215)),
        ("MACHINE-03 (cooling problem)",
         dict(product_type="L", air_temp=302.5, process_temp=310.2, rpm=1300, torque=52.0, tool_wear=90)),
        ("MACHINE-04 (power overload)",
         dict(product_type="H", air_temp=299.0, process_temp=309.5, rpm=1250, torque=72.0, tool_wear=60)),
    ]
    print("\n[PREDICT] Demo predictions on new sensor readings:")
    for machine_id, values in scenarios:
        predict_single(machine_id, **values)
