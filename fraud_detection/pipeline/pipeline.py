from fraud_detection.data_processing.loader import load_data
from fraud_detection.data_processing.features import engineer_features
from fraud_detection.model.train import train_model
from fraud_detection.model.evaluate import evaluate_model
from fraud_detection.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline():
    logger.info("Starting fraud detection pipeline")

    df = load_data()
    df = engineer_features(df)
    pipeline, X_test, y_test = train_model(df)
    metrics = evaluate_model(pipeline, X_test, y_test)

    logger.info("Pipeline finished")
    return pipeline, metrics


if __name__ == "__main__":
    run_pipeline()