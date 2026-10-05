import joblib
import pandas as pd
from functools import lru_cache

from fraud_detection.data_processing.features import engineer_features
from fraud_detection.utils.config import load_config, ROOT
from fraud_detection.utils.exceptions import ModelNotFoundError
from fraud_detection.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model():
    config = load_config()
    model_path = ROOT / config["paths"]["model_output"]

    if not model_path.exists():
        raise ModelNotFoundError(f"No trained model at {model_path}. Run training first.")

    return joblib.load(model_path)


def predict(transactions: pd.DataFrame):
    pipeline = load_model()
    features = engineer_features(transactions)

    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = pipeline.predict(features)

    logger.info(f"Scored {len(features)} transactions")
    return predictions, probabilities