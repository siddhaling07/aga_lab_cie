import numpy as np
import pandas as pd

from fraud_autoencoder.data import load_creditcard_data, prepare_training_data


def test_load_creditcard_data(tmp_path):
    sample = pd.DataFrame(
        {
            "V1": [0.1, 0.2],
            "V2": [0.3, 0.4],
            "Class": [0, 1],
        }
    )
    csv_file = tmp_path / "sample.csv"
    sample.to_csv(csv_file, index=False)

    X_scaled, y, scaler = load_creditcard_data(str(csv_file))

    assert X_scaled.shape == (2, 2)
    assert y.tolist() == [0, 1]
    assert scaler is not None

    X_train, X_test, y_test = prepare_training_data(X_scaled, y)
    assert X_train.shape[0] == 1
    assert X_test.shape == X_scaled.shape
    assert y_test.tolist() == [0, 1]
