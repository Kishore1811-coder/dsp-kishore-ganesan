import numpy as np
import pandas as pd
import joblib

from house_prices import FEATURE_COLUMNS, MODELS_DIR
from house_prices.preprocess import preprocess


def load_input(filepath):
    """Load input CSV file for inference.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Loaded DataFrame.
    """
    if not pd.io.common.file_exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    return pd.read_csv(filepath)


def load_model():
    """Load saved model from disk.

    Returns:
        Loaded model object.
    """
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def fill_missing_inference(df):
    """Fill missing values for inference data.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with NaNs filled.
    """
    from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
    df = df.copy()
    for col in CONTINUOUS_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def make_predictions(filepath):
    """Load model and return predictions for input data.

    Args:
        filepath: Path to input CSV file.

    Returns:
        Array of predicted house prices.
    """
    df = load_input(filepath)
    X = df[FEATURE_COLUMNS].copy()
    X = fill_missing_inference(X)
    X_processed = preprocess(X, is_training=False)
    model = load_model()
    predictions = model.predict(X_processed)
    return np.clip(predictions, 1, None)
