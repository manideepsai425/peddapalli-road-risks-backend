"""
train_model.py
==============
Trains an XGBoost regression model on accidents_data.csv to predict risk_score.
Run standalone: python train_model.py
Also called automatically on backend startup if model.pkl is absent.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "accidents_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "encoders.pkl")


CATEGORICAL_COLS = [
    "weather_condition", "time_of_day", "traffic_density", "road_type"
]
BOOL_COLS = ["has_intersection", "has_curve", "is_peak_hour"]
NUM_COLS = ["latitude", "longitude", "num_lanes"]


def load_and_prepare(path: str):
    df = pd.read_csv(path)

    # Boolean columns → int
    for col in BOOL_COLS:
        df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = NUM_COLS + CATEGORICAL_COLS + BOOL_COLS
    X = df[feature_cols].values
    y = df["risk_score"].values
    return X, y, encoders, feature_cols


def train():
    print("📊  Loading data from", DATA_PATH)
    X, y, encoders, feature_cols = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    print("🚀  Training XGBoost model …")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"✅  Training complete  |  MAE={mae:.4f}  R²={r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, ENCODERS_PATH)
    print(f"💾  Saved model → {MODEL_PATH}")
    print(f"💾  Saved encoders → {ENCODERS_PATH}")
    return model, encoders, feature_cols


if __name__ == "__main__":
    train()
