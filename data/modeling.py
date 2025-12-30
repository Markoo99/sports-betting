from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from src.simple_model.data_loading import load_raw_data
from src.simple_model.preprocessing import preprocess_data


FEATURE_COLUMNS = ["team_prob", "opp_prob", "spread", "total"]


def make_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix X and target vector y from the preprocessed data.
    """
    X = df[FEATURE_COLUMNS].values
    y = df["win"].values
    return X, y


def train_logistic_regression(
    X: np.ndarray, y: np.ndarray
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a logistic regression classifier and return the fitted model and metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    return model, metrics


if __name__ == "__main__":
    df_raw = load_raw_data()
    df_clean = preprocess_data(df_raw)

    print("Clean shape:", df_clean.shape)

    X, y = make_feature_matrix(df_clean)
    model, metrics = train_logistic_regression(X, y)

    print("Metrics on test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
