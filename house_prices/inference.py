import numpy as np
import pandas as pd
import joblib

from house_prices import FEATURE_COLUMNS, MODELS_DIR
from house_prices.preprocess import preprocess


def load_input(filepath):
    # load the input csv file for inference
    if not pd.io.common.file_exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    return pd.read_csv(filepath)


def load_model():
    # load the saved model from disk
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def make_predictions(filepath):
    """Load model and return predictions for input data.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Array of predicted house prices.
    """
    df = load_input(filepath)
    X = df[FEATURE_COLUMNS].copy()
    X_processed = preprocess(X, X, is_training=False)
    model = load_model()
    predictions = model.predict(X_processed)
    return np.clip(predictions, 1, None)
