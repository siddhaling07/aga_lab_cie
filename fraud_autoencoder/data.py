import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def load_creditcard_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Load credit card fraud data and scale features."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def prepare_training_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only normal transactions for training, keep all samples for evaluation."""
    X_train = X[y == 0]
    return X_train, X, y
