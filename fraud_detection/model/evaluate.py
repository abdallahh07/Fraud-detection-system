from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
)

from fraud_detection.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    metrics = {
        "pr_auc": auc(recall, precision),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Confusion matrix:\n{metrics['confusion_matrix']}")

    return metrics