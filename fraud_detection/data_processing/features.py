from fraud_detection.utils.config import load_config
from fraud_detection.utils.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df):
    config = load_config()
    drop_cols = config["features"]["drop_columns"]

    df = df.drop(columns=drop_cols)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    logger.info(f"Features ready: {list(df.columns)}")
    return df