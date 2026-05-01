import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

from house_prices import FEATURE_COLUMNS, LABEL_COLUMN, MODELS_DIR
from house_prices.preprocess import preprocess


def load_data(filepath):
    # load csv and check it exists
    if not pd.io.common.file_exists(filepath):
        raise FileNotFoundError(f"Training file not found: {filepath}")
    return pd.read_csv(filepath)


def split_data(df):
    # split into train and test before any preprocessing
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


def compute_rmsle(y_test, y_pred):
    # compute the competition metric
    y_pred = np.clip(y_pred, 1, None)
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, 2)


def train_model(X_train_processed, y_train):
    # fit a linear regression model and save it
    model = LinearRegression()
    model.fit(X_train_processed, y_train)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    return model


def build_model(filepath):
    """Train model on data and return evaluation metrics.

    Args:
        filepath: Path to training CSV file.

    Returns:
        Dictionary with metric names as keys and values as floats.
    """
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train_processed = preprocess(X_train, X_train, is_training=True)
    X_test_processed = preprocess(X_test, X_train, is_training=False)

    model = train_model(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    rmsle = compute_rmsle(y_test.values, y_pred)

    return {"rmsle": rmsle}
