import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR


def scale_features(df, is_training):
    """Scale continuous features using StandardScaler.

    Args:
        df: DataFrame with continuous features.
        is_training: If True, fit and save scaler.

    Returns:
        Scaled numpy array.
    """
    scaler_path = MODELS_DIR / "scaler.joblib"
    if is_training:
        scaler = StandardScaler()
        scaler.fit(df[CONTINUOUS_FEATURES])
        joblib.dump(scaler, scaler_path)
    else:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
    return scaler.transform(df[CONTINUOUS_FEATURES])


def encode_features(df, is_training):
    """Encode categorical features using OneHotEncoder.

    Args:
        df: DataFrame with categorical features.
        is_training: If True, fit and save encoder.

    Returns:
        Encoded numpy array.
    """
    encoder_path = MODELS_DIR / "encoder.joblib"
    if is_training:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(df[CATEGORICAL_FEATURES])
        joblib.dump(encoder, encoder_path)
    else:
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        encoder = joblib.load(encoder_path)
    return encoder.transform(df[CATEGORICAL_FEATURES])


def preprocess(df, is_training=False):
    """Preprocess features for training or inference.

    Args:
        df: Raw dataframe with feature columns.
        is_training: If True, fit and save transformers.

    Returns:
        Preprocessed numpy array ready for model.
    """
    cont_array = scale_features(df, is_training)
    cat_array = encode_features(df, is_training)
    return np.hstack([cont_array, cat_array])
