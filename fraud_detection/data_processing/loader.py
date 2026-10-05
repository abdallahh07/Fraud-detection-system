import pandas as pd

from fraud_detection.utils.config import load_config, ROOT
from fraud_detection.utils.logger import get_logger
from fraud_detection.utils.exceptions import DataLoadError

logger = get_logger(__name__)


def load_data():
    config = load_config()
    data_path = ROOT / config["data"]["raw_path"]

    if not data_path.exists():
        raise DataLoadError(f"Dataset not found at {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns")

    return df