import joblib
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection.utils.config import load_config, ROOT
from fraud_detection.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(df):
    config = load_config()

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                config["features"]["categorical"],
            )
        ],
        remainder="passthrough",
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline(
        [
            ("prep", preprocessor),
            (
                "model",
                xgb.XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=config["model"]["random_state"],
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    logger.info(f"Training on {len(X_train):,} rows")
    pipeline.fit(X_train, y_train)

    model_path = ROOT / config["paths"]["model_output"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    return pipeline, X_test, y_test