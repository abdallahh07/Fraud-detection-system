class FraudDetectionError(Exception):
    """Base exception for this project."""

class DataLoadError(FraudDetectionError):
    """Raised when the dataset can't be loaded."""

class ModelNotFoundError(FraudDetectionError):
    """Raised when the saved model file is missing."""