import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from house_prices import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR


def fill_continuous(df, ref):
    """Fill missing continuous values using median of reference set.

    Args:
        df: DataFrame to fill.
        ref: Reference DataFrame for computing medians.

    Returns:
        DataFrame with continuous NaNs filled.
    """
    df = df.copy()
    for col in CONTINUOUS_FEATURES:
        df[col] = df[col].fillna(ref[col].median())
    return df


def fill_categorical(df, ref):
    """Fill missing categorical values using mode of reference set.

    Args:
        df: DataFrame to fill.
        ref: Reference DataFrame for computing modes.

    Returns:
        DataFrame with categorical NaNs filled.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna(ref[col].mode()[0])
    return df


def scale_continuous(df, is_training: bool) -> np.ndarray:
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


def encode_categorical(df, is_training: bool) -> np.ndarray:
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


def preprocess(df, ref, is_training: bool = False) -> np.ndarray:
    """Preprocess features for training or inference.

    Args:
        df: Raw dataframe with feature columns.
        ref: Reference dataframe for imputation statistics.
        is_training: If True, fit and save transformers.

    Returns:
        Preprocessed numpy array ready for model.
    """
    df = fill_continuous(df, ref)
    df = fill_categorical(df, ref)
    cont_array = scale_continuous(df, is_training)
    cat_array = encode_categorical(df, is_training)
    return np.hstack([cont_array, cat_array])
