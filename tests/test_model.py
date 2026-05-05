import numpy as np

from fraud_autoencoder.model import build_autoencoder


def test_build_autoencoder():
    model = build_autoencoder(input_dim=5)
    random_input = np.random.random((1, 5)).astype(np.float32)
    output = model.predict(random_input)
    assert output.shape == (1, 5)
